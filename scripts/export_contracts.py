"""Export public contract JSON Schemas and the handoff examples, without inference."""

import inspect
import json
import re
from pathlib import Path

from consulting_platform import contracts
from consulting_platform.agents.mocks import fixture_chunks


def main():
    root = Path(__file__).resolve().parents[1]
    target = root / "schemas" / "v1"
    target.mkdir(parents=True, exist_ok=True)
    for name, model in inspect.getmembers(contracts, inspect.isclass):
        if (
            issubclass(model, contracts.ContractModel)
            and model.__module__ == contracts.__name__
            and name != "ContractModel"
        ):
            (target / f"{name}.json").write_text(
                json.dumps(model.model_json_schema(), indent=2) + "\n"
            )
    plan = root / "plans" / "Divpreet_Me_Agents_and_Orchestration.txt"
    examples = [
        json.loads(text) for text in re.findall(r"```json\n(.*?)\n```", plan.read_text(), re.S)
    ]
    fixtures = root / "fixtures" / "consulting"
    fixtures.mkdir(parents=True, exist_ok=True)
    (fixtures / "contract_examples.json").write_text(json.dumps(examples, indent=2) + "\n")
    (fixtures / "evidence.json").write_text(
        json.dumps([c.model_dump(mode="json") for c in fixture_chunks()], indent=2) + "\n"
    )
    print(f"Exported {len(list(target.glob('*.json')))} schemas and {len(examples)} examples.")


if __name__ == "__main__":
    main()
