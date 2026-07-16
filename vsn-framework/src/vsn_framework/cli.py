"""CLI wrapper que re-exporta la CLI de VGB."""

def main():
    import sys, os
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, os.path.join(os.path.dirname(_root), "VGB", "src"))
    from vgb.cli.main import main as vgb_main
    vgb_main()

if __name__ == "__main__":
    main()
