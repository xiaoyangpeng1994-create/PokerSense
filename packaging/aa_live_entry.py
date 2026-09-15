"""Dedicated AA monitor entry, including CLI parsing in a frozen executable."""

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    from poker_engine.desktop.aa_server import main
    main()
