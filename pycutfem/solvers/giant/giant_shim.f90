module giant_wrapper_mod
  implicit none
contains

  subroutine giant_shim(n, x, xscal, rtol, iopt, ierr, fcn, muljac)
    implicit none
    integer, intent(in) :: n
    double precision, intent(inout) :: x(n)
    double precision, intent(in) :: xscal(n)
    double precision, intent(in) :: rtol
    integer, intent(inout) :: iopt(50)
    integer, intent(out) :: ierr

    ! ---- Explicit callback prototypes (f2py needs these) ----
    interface
      subroutine fcn(n,x,f,ierr,liwku,iwku,lrwku,rwku)
        integer, intent(in) :: n
        double precision, intent(in) :: x(n)
        double precision, intent(out) :: f(n)
        integer, intent(out) :: ierr
        integer, intent(in), optional :: liwku, lrwku
        integer, intent(in), optional :: iwku(*)
        double precision, intent(in), optional :: rwku(*)
      end subroutine fcn
      subroutine muljac(n,x,v,jv,ierr,liwku,iwku,lrwku,rwku)
        integer, intent(in) :: n
        double precision, intent(in) :: x(n), v(n)
        double precision, intent(out) :: jv(n)
        integer, intent(out) :: ierr
        integer, intent(in), optional :: liwku, lrwku
        integer, intent(in), optional :: iwku(*)
        double precision, intent(in), optional :: rwku(*)
      end subroutine muljac
    end interface

    external :: fcn, muljac
    ! (No `!f2py intent(callback)` comments needed when the interface is present)

    integer :: liwk, lrwk
    integer, allocatable :: iwk(:)
    double precision, allocatable :: rwk(:)

    liwk = 1000 + 20*n
    lrwk = 5000 + 50*n
    allocate(iwk(liwk), rwk(lrwk))

    call giant(fcn, muljac, jac_stub, precon_stub, itsol_stub, &
               n, x, xscal, rtol, iopt, ierr, liwk, iwk, lrwk, rwk)

    deallocate(iwk, rwk)

  contains
    subroutine jac_stub(n, x, j, ierr, liwku, iwku, lrwku, rwku)
      integer, intent(in) :: n
      double precision, intent(in) :: x(n)
      double precision, intent(out) :: j(n,n)
      integer, intent(out) :: ierr
      integer, intent(in), optional :: liwku, lrwku
      integer, intent(in), optional :: iwku(*)
      double precision, intent(in), optional :: rwku(*)
      j = 0.0d0
      ierr = 0
    end subroutine jac_stub

    subroutine precon_stub(n, x, v, z, ierr, liwku, iwku, lrwku, rwku)
      integer, intent(in) :: n
      double precision, intent(in) :: x(n), v(n)
      double precision, intent(out) :: z(n)
      integer, intent(out) :: ierr
      integer, intent(in), optional :: liwku, lrwku
      integer, intent(in), optional :: iwku(*)
      double precision, intent(in), optional :: rwku(*)
      z = v
      ierr = 0
    end subroutine precon_stub

    subroutine itsol_stub(n, r, z, ierr, liwku, iwku, lrwku, rwku)
      integer, intent(in) :: n
      double precision, intent(in) :: r(n)
      double precision, intent(out) :: z(n)
      integer, intent(out) :: ierr
      integer, intent(in), optional :: liwku, lrwku
      integer, intent(in), optional :: iwku(*)
      double precision, intent(in), optional :: rwku(*)
      z = r
      ierr = 0
    end subroutine itsol_stub
  end subroutine giant_shim

end module giant_wrapper_mod
