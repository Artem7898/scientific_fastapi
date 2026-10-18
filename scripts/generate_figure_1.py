"""Compatibility entry point for the parameterized Python paper pipeline."""

if __package__:
    from .run_paper_pipeline import main
else:
    from run_paper_pipeline import main


if __name__ == "__main__":
    main()
