# 钢板缺陷分块分割服务 (Tiled Defect Segmentation)

FastAPI + NumPy 实现的两层 3×3 卷积缺陷分割服务。通过**分块 (tiled) 推理 +
磁盘流式 I/O**，在 **256 MiB 内存上限**的容器内处理 **16384×16384** 单通道
uint8 `.npy` 钢板巡检图，输出与整图推理**逐像素一致**的二值缺陷掩膜。

## 网络定义

| 层 | 操作 | 卷积核 | 步长 | 补零 | 激活 |
|----|------|--------|------|------|------|
| 1 | 卷积 + 偏置 | 3×3 | 2 | 整张特征图外沿补**一圈**零 | ReLU |
| 2 | 卷积 + 偏置 | 3×3 | 1 | 整张特征图外沿补**一圈**零 | — |

- 第二层得分 `score > 0` 记为 `1`，否则记为 `0`（严格大于，等于 0 记 0）。
- 全部计算使用 float32，分块路径与整图参考实现保持**完全相同的累加顺序**，
  因此结果位级一致（已在测试中用 `np.array_equal` 验证，含块缝、图边、奇数宽高）。
- 输出掩膜形状为 `(ceil(H/2), ceil(W/2))`，dtype 为 uint8，取值仅 `{0,1}`。

## 模型参数 JSON 格式

```json
{
  "layers": [
    {"kernel": [[a,b,c],[d,e,f],[g,h,i]], "bias": 0.1},
    {"kernel": [[a,b,c],[d,e,f],[g,h,i]], "bias": -0.2}
  ]
}
```

- 必须恰好包含 2 层；每层 `kernel` 为 3×3 数值数组，`bias` 为标量数值。
- 不接受 `NaN`/`Infinity`、布尔值或结构错误的参数（返回 HTTP 400）。

## 内存策略（256 MiB 如何装下 256 MiB 的图）

输入图像本身就有 `16384² = 256 MiB`，任何整体加载或 mmap 顺序扫描都会超预算。
本服务对三个大数组**一律不做内存映射**：

1. **输入 `.npy`**：仅用 `pread` 解析头部，数据按分块所需的行带读入小缓冲；
2. **第一层特征图**（float32，与输入同字节量）：写入磁盘上的标准 `.npy`，
   通过 `DiskArray` 以 `pwrite` 逐行带落盘；
3. **掩膜**（uint8，64 MiB）：同样以 `pwrite` 逐行带落盘。

第二层读取特征图时只保留当前行带 + 一行 halo。驻留内存始终只有若干分块大小的
数组（默认 `TILE_SIZE=512`，单层缓冲约 4 MiB）。

实测（16384×16384，12 核机器，单线程 BLAS）：

| 指标 | 数值 |
|------|------|
| 服务端峰值 RSS (VmHWM) | **≈ 100 MiB**（< 256 MiB） |
| `ulimit -v 262144` 硬虚存限制下 | 成功，峰值 RSS ≈ 68 MiB |
| 端到端推理耗时 | 约 4–7 s |
| 磁盘临时占用 | 特征图 256 MiB + 掩膜 64 MiB（请求结束自动清理） |

## 快速开始

### Docker Compose（推荐）

```bash
docker compose up --build
```

- 服务监听 `http://localhost:8000`
- 交互式文档与上传表单：`http://localhost:8000/docs`
- 健康检查：`GET http://localhost:8000/healthz`
- compose 已设置 `mem_limit: 256m` 硬内存上限。

### 本地运行

```bash
pip install -r requirements.txt
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  uvicorn app.main:app --host 0.0.0.0 --port 8000
```

单线程环境变量用于抑制 BLAS 线程池的额外内存占用，并保证结果确定性。

## 调用方式

### 交互式文档

浏览器打开 `/docs`，在 `POST /segment` 表单中分别上传：

- `image`：单通道 uint8、二维 `.npy` 图像文件；
- `params`：上述模型参数 JSON 文件。

### curl

```bash
curl -X POST http://localhost:8000/segment \
  -F "image=@plate.npy;type=application/octet-stream" \
  -F "params=@params.json;type=application/json" \
  --output mask.npy
```

返回体即 uint8 `.npy` 掩膜（`Content-Disposition: attachment; filename=mask.npy`）。
整张图处理完毕后才会返回响应。

### Python

```python
import httpx, numpy as np, io

with open("plate.npy", "rb") as img, open("params.json", "rb") as par:
    r = httpx.post(
        "http://localhost:8000/segment",
        files={"image": ("plate.npy", img), "params": ("params.json", par)},
        timeout=600,
    )
r.raise_for_status()
mask = np.load(io.BytesIO(r.content))  # uint8, 取值 {0,1}
```

下载大掩膜时建议用 `httpx.stream(...)` 流式写盘，避免客户端侧缓冲整个响应。

## 错误处理

| 情况 | 状态码 |
|------|--------|
| 图像不是合法 `.npy` / dtype 非 uint8 / 非二维 / 为空 | 400 |
| 参数 JSON 非法、结构不符、含 NaN/Infinity | 400 |
| 上传体积超过 `MAX_UPLOAD_BYTES`（默认 2 GiB） | 413 |
| 推理过程发生未预期错误 | 500 |

所有错误返回 `{"detail": "..."}`，且对应的临时目录都会被清理。

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `TILE_SIZE` | `512` | 每层输出网格的分块边长（输出像素） |
| `TMPDIR` | `/tmp` | 上传缓冲与中间 `.npy` 的临时目录（应在磁盘上，勿挂 tmpfs） |
| `MAX_UPLOAD_BYTES` | `2147483648` | 单文件上传大小上限 |
| `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` | `1` | 单线程 BLAS，控制内存与确定性 |

## 测试

```bash
pip install pytest httpx
pytest -q
```

测试覆盖：

- 分块推理 vs 整图参考在 1×1 到 257×255、奇偶尺寸、不同分块大小下逐像素相等；
- 图边/角的零填充行为、严格 `>0` 阈值、单热像素感受野；
- 参数与输入的全部校验分支；
- FastAPI 端到端上传、错误码、`/docs` 可用性。

整图参考实现见 `app/inference.py::reference_full_inference`。
全尺寸内存基准：`BENCH_SIZE=16384 python -I scripts/bench_large.py <数据目录>`。

## 项目结构

```
app/
  inference.py   # DiskArray 磁盘分块数组、两层 tiled 卷积、整图参考实现
  main.py        # FastAPI 路由、上传流式落盘、校验与错误处理
tests/           # 正确性与 HTTP 端到端测试
scripts/
  bench_large.py # 大尺寸内存/耗时基准
Dockerfile
docker-compose.yml
requirements.txt
```
