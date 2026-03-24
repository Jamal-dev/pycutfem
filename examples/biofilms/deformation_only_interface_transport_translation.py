import sys

from examples.biofilms.deformation_only_interface_transport import main


if __name__ == "__main__":
    if "--case" not in sys.argv:
        sys.argv.extend(["--case", "translation"])
    main()
