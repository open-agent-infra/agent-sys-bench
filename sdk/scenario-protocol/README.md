# scenario-protocol

Defines the wire protocol between **scenario loadgens** and **agent services**.

The loadgen is a pure HTTP client — it imports no agent code. It constructs
scenario-defined `Request` models from a manifest, then dispatches them via
`ScenarioClient` to the agent's fast-service endpoint. The agent side
documents which RPC endpoint names it serves for each scenario.

## Usage

```python
from scenario_protocol import ScenarioClient
from agsbench_request.swe.interface import SWETaskRequest, SWETaskResult, RPC_NAME

client = ScenarioClient(endpoint="http://mini-sweagent-service:1801")
req = SWETaskRequest(instance_id="django__django-16379", instance={...})
resp: SWETaskResult = client.dispatch(RPC_NAME, req, response_type=SWETaskResult)
```
