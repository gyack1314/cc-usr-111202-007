# 生物标志物双阈值分流的风险覆盖前沿

这是研究资料维护用的独立Python项目。`data/`保存合成记录，`research_data.io`负责严格JSON读取与SHA-256来源摘要，`research_data.domain`提供当前原始资料统计，`research_data.cli`将摘要输出为JSON。字段、时间、计数和量纲见[资料约定](docs/data-contract.md)。所有研究、中心及受试者均为虚构。

Python 3.11及以上版本可直接使用，当前没有第三方依赖。在项目根目录运行：

```sh
python -m unittest discover -s tests -v
python -m compileall -q research_data tests
python -m research_data.cli inspect data/normal.json
```

双阈值分流分析(development选点、holdout核验):

```sh
python -m research_data.cli triage data/normal.json
python -m research_data.cli triage data/normal.json --max-pairs 100000
```

`triage`在拆分前冻结的候选阈值与风险上限下,枚举全人群共享的(lo,hi)规则,输出可行规则的覆盖率-损失Pareto前沿、固定目标(覆盖率最高,其次损失最低,其次阈值字典序最小)的选点及holdout核验;`--max-pairs`资源上限触发时`frontier_complete`为false,不声称前沿完整。口径见[资料约定](docs/data-contract.md)。

`inspect`只显示已有摘要，不修改来源。文件名以`invalid-`起始的资料包含故意不合法的输入，其他文件是同结构的业务边界。数据摘要与来源指纹应同时保留，原始JSON不覆盖。

可使用固定镜像容器：`docker build -t research-material .`，随后`docker run --rm research-material`查看资料；`docker run --rm research-material python -m unittest discover -s tests -v`运行现有检查。全部资料在项目内，运行不需要外部账号或服务。
