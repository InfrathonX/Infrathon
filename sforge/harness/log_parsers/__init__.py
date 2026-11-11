from sforge.harness.log_parsers.pytest_v import parse_pytest_v
from sforge.harness.log_parsers.score_sum import parse_score_sum
from sforge.harness.log_parsers.structured_json import parse_structured_json

MAP_TASK_TO_PARSER: dict[str, callable] = {
    "pytest_v": parse_pytest_v,
    "score_sum": parse_score_sum,
    "structured_json": parse_structured_json,
}


def get_parser(parser_name: str) -> callable:
    if parser_name not in MAP_TASK_TO_PARSER:
        raise ValueError(
            f"Unknown parser: {parser_name}. Available: {list(MAP_TASK_TO_PARSER.keys())}"
        )
    return MAP_TASK_TO_PARSER[parser_name]
