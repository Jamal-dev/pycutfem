import sys

from examples.biofilms.deformation_only_interface_transport import main


if __name__ == "__main__":
    sys.argv.extend(["--case", "rotation"])
    main()
