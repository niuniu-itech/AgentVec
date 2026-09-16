"""Command-line entry points for generation, validation and schedule replay."""
import argparse
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(prog="agentvec", description="Validated RVV migration tools")
    parser.add_argument("command", choices=["migrate", "dag", "ascend", "difftest", "schedules"])
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in ("-h", "--help"):
        parser.print_help()
        return 0
    args = parser.parse_args(arguments[:1])
    if args.command == "migrate":
        from .pipeline import main as execute
    elif args.command == "dag":
        from .dag_pipeline import main as execute
    elif args.command == "ascend":
        from .ascend.pipeline import main as execute
    elif args.command == "difftest":
        from .difftest import main as execute
    else:
        from .schedules import main as execute
    return execute(arguments[1:])
