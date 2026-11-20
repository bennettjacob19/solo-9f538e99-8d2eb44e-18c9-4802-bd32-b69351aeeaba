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
- 急件抢占同样在该 256 MiB 上限内完成：任一时刻只有一张图占用计算位，
  让位发生在块边界且该块缓冲已释放。

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
- `params`：上述模型参数 JSON 文件；
- `urgent`（可选，默认未勾选）：急件开关，见下节「急件（复拍小图）」。

### curl

```bash
curl -X POST http://localhost:8000/segment \
  -F "image=@plate.npy;type=application/octet-stream" \
  -F "params=@params.json;type=application/json" \
  --output mask.npy
```

返回体即 uint8 `.npy` 掩膜（`Content-Disposition: attachment; filename=mask.npy`）。
整张图处理完毕后才会返回响应。

### 急件（复拍小图）

`POST /segment` 表单新增可选字段 `urgent`（布尔，默认 `false`；原调用方式不变）：

```bash
curl -X POST http://localhost:8000/segment \
  -F "image=@reshot.npy;type=application/octet-stream" \
  -F "params=@params.json;type=application/json" \
  -F "urgent=true" \
  --output mask.npy
```

- **尺寸限制**：急件的宽、高各不得超过 **1024 像素**（`H <= 1024` 且 `W <= 1024`），
  超限返回 HTTP 400（`detail` 说明实际尺寸）；校验在上传落盘后、进入计算前完成，
  被拒绝的急件不会中断或拖慢正在运行的常规单。
- **抢占时机**：当常规单尚未让过位、仍有块未算，且已有校验通过的急件在排队时，
  常规单写完当前块、释放该块计算缓冲后立刻交棒；两层（第一层与第二层）都以
  块边界作为交棒点，最后一层最后一块写完后掩膜已完整，不再交棒。
- **接手顺序**：急件按受理顺序（FIFO）依次独占计算位；常规单在交棒瞬间已受理的
  急件全部完成后**立刻续算**，沿用磁盘上已写好的中间特征图与掩膜块，不重算已有块。
- **让位上限**：每张常规单中途最多让位一次，之后到达的急件排在它后面。
- **失败处理**：急件计算失败时该请求返回错误（400/500），常规单照常续算不受影响。
- **一致性**：任一时刻只有一张图占用计算位；交棒只发生在块边界，中间结果全部
  落盘，因此常规单续算与一次跑完、以及各急件单独运行，结果均**逐像素一致**。

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
| 急件（`urgent=true`）宽或高超过 1024 像素 | 400 |
| `urgent` 字段不是合法布尔值 | 422 |
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
- 急件尺寸限制、`/docs` 表单字段、原上传方式兼容；
- 单计算位互斥、急件按受理顺序接手、常规单最多让位一次、交棒后从已写块续算、
  急件失败不阻塞常规单、超限急件不打断常规单（含并发真实推理端到端）；
- 在每个块边界调用交棒钩子后结果仍与整图逐像素相等；
- FastAPI 端到端上传、错误码、`/docs` 可用性。

整图参考实现见 `app/inference.py::reference_full_inference`。
全尺寸内存基准：`BENCH_SIZE=16384 python -I scripts/bench_large.py <数据目录>`。

## 项目结构

```
app/
  inference.py   # DiskArray 磁盘分块数组、两层 tiled 卷积、整图参考实现
  scheduler.py   # 单计算位调度：急件抢占、常规单让位一次后原地续算
  main.py        # FastAPI 路由、上传流式落盘、校验与错误处理
tests/           # 正确性、调度策略与 HTTP 端到端测试
scripts/
  bench_large.py # 大尺寸内存/耗时基准
Dockerfile
docker-compose.yml
requirements.txt
```
