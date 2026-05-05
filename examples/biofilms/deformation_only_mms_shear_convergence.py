import sys

from examples.biofilms.deformation_only_mms_convergence import main


if __name__ == "__main__":
    if "--case" not in sys.argv:
        sys.argv.extend(["--case", "shear"])
    main()
