      SUBROUTINE GIANT(N,FCN,JAC,X,XSCAL,RTOL,IOPT,IERR,
     $LIWK,IWK,LRWK,RWK,LIWKU,IWKU,LRWKU,RWKU,MULJAC,PRECON,ITSOL)
C*    Begin Prologue GIANT
      INTEGER N
      EXTERNAL FCN,JAC
      DOUBLE PRECISION X(N),XSCAL(N)
      DOUBLE PRECISION RTOL
      INTEGER IOPT(50)
      INTEGER IERR
      INTEGER LIWK
      INTEGER IWK(LIWK)
      INTEGER LRWK
      DOUBLE PRECISION RWK(LRWK)
      INTEGER LIWKU
      INTEGER IWKU(LIWKU)
      INTEGER LRWKU
      DOUBLE PRECISION RWKU(LRWKU)
      EXTERNAL MULJAC,PRECON,ITSOL
C   --------------------------------------------------------------
C
C*  Title
C
C     Numerical solution of large scale highly nonlinear systems with
C     Global (G) Inexact (I) Affine-invariant (A) Newton (N)
C     Techniques (T)
C
C*  Written by        U. Nowak, L. Weimann 
C*  Purpose           Solution of large scale systems of highly
C                     nonlinear equations
C*  Method            Damped affine invariant Newton method combined
C                     with iterative solution of arising linear systems  
C                     (see references below)
C*  Category          F2a. - Systems of nonlinear equations
C*  Keywords          Nonlinear equations, large systems,
C                     inexact Newton methods, iterative methods
C*  Version           2.3
C*  Revision          September 1991
C*  Latest Change     July 2000
C*  Library           CodeLib
C*  Code              Fortran 77, Double Precision
C*  Environment       Standard Fortran 77 environment on PC's,
C                     workstations and hosts.
C*  Copyright     (c) Konrad-Zuse-Zentrum fuer
C                     Informationstechnik Berlin (ZIB)
C                     Takustrasse 7, D-14195 Berlin-Dahlem
C                     phone : + 49/30/84185-0
C                     fax   : + 49/30/84185-125
C*  Contact           Bodo Erdmann
C                     ZIB, Division Scientific Computing, 
C                          Department Numerical Analysis and Modelling
C                     phone : + 49/30/84185-185
C                     fax   : + 49/30/84185-107
C                     e-mail: erdmann@zib.de
C
C*  References 
C
C     /1/ P. Deuflhard:
C         Newton Methods for Nonlinear Problems. -
C         Affine Invariance and Adaptive Algorithms.
C         Series Computational Mathematics 35, Springer (2004)
C
C     /2/ P. Deuflhard:
C         Global Inexact Newton Methods for Very Large Scale
C         Nonlinear Problems.
C         ZIB, Preprint SC 90-2 (February 1990)
C
C     /3/ P. Deuflhard, R. Freund, A. Walter:
C         Fast Secant Methods for the Iterative Solution of
C         Large Nonsymmetric Linear Systems.
C         ZIB, Preprint SC 90-5 (July 1990)
C
C     /4/ U. Nowak, L. Weimann:
C         GIANT - A Software Package for the Numerical Solution
C         of Very Large Systems of Highly Nonlinear Equations. 
C         ZIB, Technical Report TR 90-11 (December 1990)
C
C  ---------------------------------------------------------------
C
C* Licence
C    You may use or modify this code for your own non commercial
C    purposes for an unlimited time. 
C    In any case you should not deliver this code without a special 
C    permission of ZIB.
C    In case you intend to use the code commercially, we oblige you
C    to sign an according licence agreement with ZIB.
C
C* Warranty 
C    This code has been tested up to a certain level. Defects and
C    weaknesses, which may be included in the code, do not establish
C    any warranties by ZIB. ZIB does not take over any liabilities
C    which may follow from aquisition or application of this code.
C
C* Software status 
C    This code is under care of ZIB and belongs to ZIB software class 1.
C
C  ---------------------------------------------------------------
C
C*    Summary:
C     ========
C
C     Damped Newton-algorithm with inexact Newton techniques for
C     large systems of highly nonlinear equations - 
C     damping strategy due to Ref. /1/,/2/
C     accuracy matching strategy due to Ref. /2/
C
C     (The iteration is done by subroutine NJINT actually. GIANT
C      itself does some house keeping and builds up workspace.)
C
C     The numerical solution of the arising linear equations is
C     done by a special implementation of the iterative solver
C     'Good Broyden' - see Ref. /3/ - or some user 
C     supplied iterative solver (subroutine parameter ITSOL).
C
C     This is a driver routine for the core solver NJINT.
C
C  ---------------------------------------------------------------
C
C*    Parameters list description (* marks inout parameters)
C     ======================================================
C
C*    External subroutines (to be supplied by the user)
C     =================================================
C 
C     (Caution: Arguments declared as (input) must not
C               be altered by the user subroutines ! )
C
C       FCN(N,X,F,RWKU,IWKU,NFCN,IFAIL)
C                 Ext    Problem function subroutine
C         N         Int    Number of vector components (input)
C         X(N)      Dble   Vector of unknowns (input)
C         F(N)      Dble   Vector of problem function values (output)
C       * RWKU(*)   Dble   Real Workspace for the user routines -
C                          passed from the parameter RWKU of the
C                          driver subroutine GIANT.
C       * IWKU(*)   Int    Integer Workspace for the user routines -
C                          passed from the parameter IWKU of the
C                          driver subroutine GIANT.
C         NFCN      Int    Count of FCN calls. (input)
C                          Must not be altered by FCN.
C         IFAIL     Int    FCN evaluation-failure indicator. (output)
C                          Has always value 0 (zero) on input.
C                          Indicates failure of FCN evaluation
C                          and causes termination of GIANT,
C                          if set to a negative value on output
C
C
C       JAC(FCN,N,X,XSCAL,F,RWKU,IWKU,NJAC,IFAIL) 
C                  Ext    Jacobian matrix subroutine
C         FCN        Ext    The problem function (subroutine) 
C                           reference
C         N          Int    Number of vector components (input)
C         X(N)       Dble   Vector of unknowns (input)
C         XSCAL(N)   Dble   Array of scaling values associated to X(N)
C                           (input)
C         F(N)       Dble   Vector of problem function values as 
C                           supplied by subroutine FCN with same input
C                           argument X(N) (input)
C       * RWKU(*)    Dble   Real Workspace for the user routines -
C                           passed from the parameter RWKU of the
C                           driver subroutine GIANT.
C       * IWKU(*)    Int    Integer Workspace for the user routines -
C                           passed from the parameter IWKU of the
C                           driver subroutine GIANT.
C         NJAC       Int    Count of JAC calls. (input)
C                           Must not be altered by JAC.
C         IFAIL      Int    JAC evaluation-failure indicator. (output)
C                           Has always value 0 (zero) on input.
C                           Indicates failure of JAC evaluation
C                           and causes termination of GIANT,
C                           if set to a negative value on output
C
C
C       MULJAC( N, X, Y, RWKU, IWKU )
C                Ext    Jacobian * vector product subroutine
C         N        Int    The number of vector components (input)
C         X(N)     Dble   The vector to be multiplied by the Jacobian
C                         (input)
C         Y(N)     Dble   The array to get the result vector
C                         Jacobian * X (output)
C       * RWKU(*)  Dble   Real Workspace for the user routines -
C                         passed from the parameter RWKU of the
C                         driver subroutine GIANT.
C       * IWKU(*)  Int    Integer Workspace for the user routines -
C                         passed from the parameter IWKU of the
C                         driver subroutine GIANT.
C         
C         
C       PRECON( N, R, Z, RWKU, IWKU )
C                Ext    Preconditioning solver subroutine for system
C                       M * Z = R
C         N        Int    The number of vector components (input)
C         R(N)     Dble   The right hand side of the system (input)
C         Z(N)     Dble   The array to get the solution vector (output)
C       * RWKU(*)  Dble   Real Workspace for the user routines -
C                         passed from the parameter RWKU of the
C                         driver subroutine GIANT.
C       * IWKU(*)  Int    Integer Workspace for the user routines -
C                         passed from the parameter IWKU of the
C                         driver subroutine GIANT.
C
C
C       ITSOL( N, B, X, XDEL, XSCAL, MULJAC, PRECON,
C      1       TOL, ITMAX, ITER, ERR, IERR, IOPT,
C      2       LRWK, RWK, NRW, LIWK, IWK, NIW, LRWKU, RWKU, LIWKU, IWKU )
C                   Ext    Iterative linear solver subroutine
C         N           Int    The number of vector components (input)
C         B(N)        Dble   The right hand side of the system (input)
C                            Must not be altered by ITSOL !
C       * X(N)        Dble   The array to get the approximate solution
C                            vector (output)
C                            On input: the startiterate
C       * XDEL(N)     Dble   Must get on output the difference between
C                            the start- and the final iterate (in case
C                            of continuation iteration (IOPT(1).EQ.1):
C                            add the additional difference)
C                            On input supplied by GIANT with the zero
C                            vector, if a new iteration starts, and
C                            with the latest output from ITSOL, if an
C                            iteration will be continued.
C       * XSCAL(N)    Dble   Array of scaling values for the solution 
C                            vector (may be altered)
C                            On input: The current scaling values of 
C                            Newton iterate.
C         MULJAC      Ext    Name of Jacobian*vector product subroutine
C         PRECON      Ext    Name of preconditioner subroutine
C         TOL         Dble   Prescribed tolerance for the convergence 
C                            criterium. (input)
C         ITMAX       Int    Maximum number of iterations allowed 
C                            (input).
C         ITER        Int    Number of iterations done to get the
C                            solution (output).
C         ERR         Dble   Error estimate of error in final 
C                            approximate solution. (output)
C         IERR        Int    Return error flag. Zero indicates, that
C                            no error occured. (output)
C       * IOPT(50)    Int    Linear solver options array as supplied by
C                            GIANT. (input)
C                            Information about the type of stopping
C                            criterion actually activated should be
C                            stored to position IOPT(50) (on output)
C                            See below for details.
C         LRWK        Int    Length of workspace RWK 
C       * RWK(LRWK)   Dble   Workspace for use by ITSOL
C         NRW         Int    Amount of workspace RWK used (output) 
C         LIWK        Int    Length of workspace IWK 
C       * IWK(LIWK)   Int    Workspace for use by ITSOL
C         NIW         Int    Amount of workspace IWK used  (output) 
C         LRWKU       Int    Length of workspace RWKU 
C       * RWKU(LRWKU) Dble   Real Workspace for the user routines -
C                            passed from the parameter RWKU of the
C                            driver subroutine GIANT.
C                            - Not intended for use by ITSOL
C         LIWKU       Int    Length of workspace IWKU
C       * IWKU(LIWKU) Int    Integer Workspace for the user routines -
C                            passed from the parameter IWKU of the
C                            driver subroutine GIANT.
C                            - Not intended for use by ITSOL
C
C     Settings for options array IOPT of ITSOL as supplied by GIANT:
C     (used positions must not be altered by ITSOL - except IOPT(50) -
C      see below)
C     --------------------------------------------------------------
C     Pos. Name            Meaning
C
C       1  QSUCC           Indicator for the iteration modus:
C                          =0 : A new iteration will be started
C                          =1 : A previously terminated iteration will 
C                               be continued - recommended to be done
C                               without some special restart.
C                               (usally with a smaller tolerance RTOL
C                                prescribed as before).
C       2..12              Reserved
C      13  MPRLIN          Output level for the iterative linear solver 
C                          monitor
C                          Is recommended to be interpreted as follows:
C                          = 0 : no output will be written
C                          = 1 : only a summary output will be written
C                          > 1 : reserved for future use
C                          =-j : A summary output and additionally each 
C                                j-th iterates statistics will be 
C                                written
C      14  LULIN           Logical unit number for print monitor
C      15..16              Reserved
C      17  MPROPT          Print monitor option:
C                          Is recommended to be interpreted as follows:
C                          = 0: Standard print monitor
C                          = 1: Test print monitor for special purposes
C      18                  Reserved
C      19  MPRTIM          Output level MPRTIM for the time monitor.
C                          See IOPT(19) of driver subroutine GIANT 
C                          and see also note 0 below.
C      20..21              Reserved
C      22  LUPLO           Logical output unit for special information
C                          (The value zero should be interpreted to do
C                           no output)
C      23..30              Reserved
C      31  KMAX            Maximum number of latest iterates to be saved
C                          and used by the iterative linear solver Good
C                          Broyden. May be interpreted by a user
C                          supplied linear solver in a simular way.
C                          See IOPT(41) in driver subroutine GIANT for 
C                          details.
C      32                  Reserved
C      33  IFINI           Type of stop criterium required to be 
C                          satisfied by ITSOL:
C                          =1: stop, if relcorr(X) < TOL
C                          =2: stop, if relcorr(X) < TOL or 
C                                    if relcorr(XDEL) < 2*TOL
C      34..49              Reserved
C      50  ITERM           On output, this field should contain the
C                          information, which criterium leaded to 
C                          convergence stop  (if on input 2):
C                          =1: if relcorr(X) < EPS has been satisfied
C                          =2: relcorr(DEL) < 2*EPS has been satisfied
C
C     Note 0.
C        The time monitor may be used in connection with a user supplied
C        linear solver in the same way as with the solver Good Broyden.
C        Assuming, that the Jacobian times vector and preconditioner
C        subroutines are named MULJAC and PRECON, you have to modify
C        the linear solvers code as outlined below:
C
C        SUBROUTINE your-itsol(....,IOPT,....)
C        ...
C        ...
C  C   Insert following line
C        MPRTIM=IOPT(8)
C        ...
C        ...
C  C   Insert following line
C        IF (MPRTIM.NE.0) CALL MONON (5)
C  C
C        CALL MULJAC(...)
C  C   Insert following line
C        IF (MPRTIM.NE.0) CALL MONOFF (5)
C        ...
C        ...
C  C   Insert following line
C       IF (MPRTIM.NE.0) CALL MONON (3)
C  C
C       CALL PRECON(...)
C  C   Insert following line
C       IF (MPRTIM.NE.0) CALL MONOFF (3)
C        ...
C        ...
C       END       
C
C     Settings for first positions of workspace array RWK of ITSOL
C     as supplied by GIANT for iterative solver Good Broyden:
C     (may be overwritten by ITSOL)
C     --------------------------------------------------------------
C     Pos.    Description
C       1     A security factor, which uses Good Broyden to multiply
C             the raw estimated error with it. Passed from RWK(41)
C             of the GIANT subroutine for computations of the ordinary
C             Newton correction and passed from RWK(42) of GIANT for 
C             computations of the simplified Newton correction.
C       2     Only meaningfull for Good Broyden - see RWK(43) of GIANT.
C       3     Only meaningfull for Good Broyden - see RWK(44) of GIANT.
C       4     Only meaningfull for Good Broyden - see RWK(45) of GIANT.
C
C
C*    Input parameters of GIANT
C     =========================
C
C     N              Int    Number of unknowns
C   * X(N)           Dble   Initial estimate of parameters
C   * XSCAL(N)       Dble   User scaling (lower threshold) of the 
C                           iteration vector X(N)
C   * RTOL           Dble   Required relative precision of
C                           solution components -
C                           RTOL.GE.EPMACH*TEN*N
C                           See also note 1a below.
C   * IOPT(50)       Int    Array of run-time options. Set to zero
C                           to get default values (details see below)
C
C     Note 1a. Be careful with your choice of RTOL:
C              stringend RTOL requirements (1.0d-6 till 1.0d-12) are
C              in general uncritical for standard Newton methods, but
C              may at least increase the amount of work for the
C              iterative linear solution drastically.
C
C*    Output parameters of GIANT
C     ==========================
C
C   * X(N)           Dble   Solution values ( or final values,
C                           respectively )
C   * XSCAL(N)       Dble   After return with IERR.GE.0, it contains
C                           the latest internal scaling vector used
C                           After return with IERR.EQ.-1 in onestep-
C                           mode it contains a possibly adapted 
C                           (as described below) user scaling vector:
C                           If (XSCAL(I).LT. SMALL) XSCAL(I) = SMALL ,
C                           If (XSCAL(I).GT. GREAT) XSCAL(I) = GREAT .
C                           For SMALL and GREAT, see section machine
C                           constants below  and regard note 1b.
C   * RTOL           Dble   Finally achieved accuracy
C   * IOPT(50)       Int    1. IOPT- fields with nonzero default values, 
C                              which were zero on input, are set to the
C                              default value.
C                           2. If stepwise mode was selected 
C                              (IOPT(2)=1), IOPT(1) is prepared for a
C                              successive call, e.g. set to one.
C                           
C     IERR           Int    Return value parameter
C                           =-1 sucessfull completion of one iteration
C                               step, subsequent iterations are needed 
C                               to get a solution. (stepwise mode only) 
C                           = 0 successfull completion of the iteration,
C                               solution has been computed
C                           > 0 see list of error messages below
C
C     Note 1b.
C        The machine dependent values SMALL, GREAT and EPMACH are
C        gained from calls of the machine constants function ZIBCONST.
C        As delivered, this function is adapted to use constants 
C        suitable for all machines with IEEE arithmetic. If you use
C        another type of machine, you may change ZIBCONST.
C
C*    Workspace parameters of GIANT
C     =============================
C
C     LIWK           Int    Declared dimension of integer workspace.
C                           Required minimum:
C                           For standard linear system solver: 50
C                           For user supplied linear solver ITSOL:
C                           50 + (integer workspace needed by ITSOL)
C  *  IWK(LIWK)      Int    Integer Workspace
C     LRWK           Int    Declared dimension of real workspace.
C                           Required minimum:
C                           For standard linear system solver:
C                           (11+KMAX)*N+2*KMAX+77 ,
C                           (see internal parameter KMAX = IOPT(41) 
C                            below, default is 9)
C                           For user supplied linear solver ITSOL:
C                           7*N+60 + (real workspace needed by ITSOL)
C   * RWK(LRWK)      Dble   Real Workspace
C     LIWKU          Int    Declared dimension of user problem function
C                           integer workspace.
C     IWKU(LIWKU)    Int    User integer workspace - passed to the
C                           user subroutines. Intended for keeping
C                           integer information concerning the users
C                           problem function, it's Jacobian and the
C                           preconditioner.
C     LRWKU          Int    Declared dimension of user problem function 
C                           real workspace.
C     RWKU(LRWKU)    Dble   User real workspace - passed to the
C                           user subroutines. Intended for keeping
C                           floating point information concerning the
C                           users problem function, it's Jacobian and
C                           the preconditioner.
C
C     Note 2a.  A test on sufficient workspace is made. If this
C               test fails, IERR is set to 10 and an error-message
C               is issued from which the minimum of required
C               workspace size can be obtained.
C
C     Note 2b.  The first 50 elements of IWK and RWK are partially 
C               used as input for internal algorithm parameters (for
C               details, see below). In order to set the default values
C               of these parameters, the fields must be set to zero. 
C               Therefore, it's recommended always to initialize the
C               first 50 elements of both workspaces to zero before the
C               initial call.
C
C*   Options IOPT:
C    =============
C
C     Pos. Name   Default  Meaning
C
C       1  QSUCC  0        =0 (.FALSE.) initial call:
C                             GIANT is not yet initialized, i.e. this is
C                             the first call for this nonlinear system.
C                             At successfull return with MODE=1,
C                             QSUCC is set to 1.
C                          =1 (.TRUE.) successive call:
C                             GIANT is initialized already and is now
C                             called to perform one or more following
C                             Newton-iteration steps.
C                             ATTENTION:
C                                Don't destroy the contents of
C                                IWK(i) for 50 < i < NIWKFR and
C                                RWK(k) for 50 < k < NRWKFR
C                                and of fields of types other than IN
C                                (of the first 50 elements)
C                                before successive calls.
C       2  MODE   0        =0 Standard mode initial call:
C                             Return when the required accuracy for the
C                             iteration vector is reached. User defined
C                             parameters are evaluated and checked.
C                             Standard mode successive call:
C                             If GIANT was called previously with MODE=1,
C                             it performs all remaining iteration steps.
C                          =1 Stepwise mode:
C                             Return after one Newton iteration step.
C       3..7               Reserved 
C       8  LTYP   0        = 0 The default linear solver is used 
C                              (see =1)
C                          = 1 The iterative solver 'good Broyden'
C                              is used
C                          = 9 A user supplied iterative solver
C                              (given by subroutine ITSOL) is used
C       9  ISCAL  0        Determines how to scale the iterate-vector:
C                          =0 The user supplied scaling vector XSCAL is
C                             used as a (componentwise) lower threshold
C                             of the actual scaling vector
C                          =1 The vector XSCAL is always used as the
C                             actual scaling vector
C      10                  Reserved
C      11  MPRERR 0        Print error messages
C                          =0 No output
C                          =1 Error messages
C                          =2 Warnings additionally
C                          =3 Informal messages additionally
C      12  LUERR  6        Logical unit number for error messages
C      13  MPRMON 0        Print iteration Monitor
C                          =0 No output
C                          =1 Standard output
C                          =2 Summary iteration monitor additionally
C                          =3 Detailed iteration monitor additionally
C                          =4,5,6 Outputs with increasing level addi-
C                             tional increasing information for code
C                             testing purposes. Level 6 produces
C                             in general extremely large output!
C      14  LUMON  6        Logical unit number for iteration monitor
C      15  MPRSOL 0        Print solutions
C                          =0 No output
C                          =1 Initial values and solution values
C                          =2 Intermediate iterates additionally
C      16  LUSOL  6        Logical unit number for solutions
C      17  MPRLIN 0        Output level for the iterative linear solver
C                          monitor
C                          Is interpreted by Good Broyden as follows:
C                          = 0 : no output will be written
C                          = 1 : only a summary output will be written
C                          > 1 : reserved for future use
C                          =-j : A summary output and additionally 
C                                each j-th iterates statistics will 
C                                be written to the monitor.
C                          Hint: if you choose MPRLIN=-j with j > LITMAX
C                          (see IWK(41)=LITMAX), only the start iterate
C                          and special messages generated during the 
C                          linear solvers iteration will be written.
C      18  LULIN  6        Logical output unit for the iterative linear 
C                          solver monitor
C      19  MPRTIM 0        Output level for the time monitor
C                          = 0 : no time measurement and no output
C                          = 1 : time measurement will be done and
C                                summary output will be written -
C                                regard note 4a.
C      20  LUTIM  6        Logical output unit for time monitor
C      21..30              Reserved
C      31  NONLIN 3        Problem type specification
C                          =1 Linear problem
C                             Warning: If specified, no check will be
C                             done, if the problem is really linear, and
C                             GIANT terminates unconditionally after one
C                             Newton-iteration step.
C                          =2 Mildly nonlinear problem
C                          =3 Highly nonlinear problem
C                          =4 Extremely nonlinear problem
C      32..37              Reserved
C      38  IBDAMP          Bounded damping strategy switch:
C                          =0 The default switch takes place, dependent
C                             on the setting of NONLIN (=IOPT(31)):
C                             NONLIN = 0,1,2,3 -> IBDAMP = off ,
C                             NONLIN = 4 -> IBDAMP = on
C                          =1 means always IBDAMP = on 
C                          =2 means always IBDAMP = off
C      39..40              Reserved
C      41  KMAX   9        Maximum number of latest iterates to be saved
C                          by the iterative linear solver Good Broyden.
C                          Values <=0 will be special handled as listed
C                          below:
C                          a. An input <=-2 means KMAX=0 will be used. 
C                          b. A "-1" input means, that there no limit 
C                             applies, e.g. the Good Broyden will be
C                             served with the maximum possible value 
C                             allowed by the total workspace amount (but
C                             a value < 2 will not be accepted!)
C                             On output, KMAX will be set to the compu-
C                             ted value, which has (or would have) been
C                             accepted
C                          c. A zero input means KMAX will be set to the
C                             default value 9.
C      42..45              Reserved
C      46..50              User options (see note 4b)
C
C     Note 3:
C         If GIANT terminates with IERR=2 (maximum iterations)
C         or  IERR=3 (small damping factor), you may try to continue
C         the iteration by increasing NITMAX or decreasing FCMIN
C         (see RWK) and setting QSUCC to 1.
C
C     Note 4a:
C        The integrated time monitor calls the machine dependent
C        subroutine SECON to get the actual time stamp in form
C        of a real number (Single precision). As delivered, this
C        subroutine always return 0.0 as time stamp value. Refer
C        to the compiler- or library manual of the FORTRAN compiler
C        which you actually use to find out how to get the actual
C        time stamp on your machine.
C
C     Note 4b:
C         The user options may be interpreted by the user replacable
C         routine NJSOUT  - the distributed version
C         of NJSOUT actually uses IOPT(46) as follows:
C         0 = standard plotdata output (may be postprocessed by a user-
C             written graphical program)
C         1 = plotdata output is suitable as input to the graphical
C             package GRAZIL (based on GKS), which has been developed
C             at ZIB. 
C
C
C*   Optional INTEGER input/output in IWK:
C    =======================================
C
C     Pos. Name          Meaning
C
C      1   NITER  IN/OUT Number of Newton-iterations
C      2                 Reserved
C      3   NCORR  IN/OUT Number of corrector steps
C      4   NFCN   IN/OUT Number of FCN-evaluations
C      5   NJAC   IN/OUT Number of JAC-evaluations
C      6   NLINOR IN/OUT Total number of linear solver iterations done
C                        for computing ordinary Newton corrections 
C      7   NLINSI IN/OUT Total number of linear solver iterations done
C                        for computing simplified Newton corrections
C      8   NAMCF  IN/OUT Count of violations of the accuracy matching 
C                        condition for the hk computation. Possible
C                        reason of violations is an inappropriate error
C                        estimate by the iterative linear solver. 
C                        Possible help: increase at least RHOORD ( see
C                        RWK(41) ), or use other iterative linear solver
C      9   NMULJ  IN/OUT Number of (pairwise) calls of MULJAC and PRECON
C                        (only, if used with Good Broyden)
C     10                 A user programmable counter returned by the 
C                        iterative linear solver through IOPT(48)
C     11                 A user programmable counter returned by the 
C                        iterative linear solver through IOPT(49)
C     12   IDCODE IN/OUT Output: The 6 decimal digits program identi-
C                        fication number ppvvvv, consisting of the
C                        program code pp and the version code vvvv.
C                        Input: If containing a negative number,
C                        it will only be overwritten by the identi-
C                        fication number, immediately followed by
C                        a return to the calling program.      
C     13                 Reserved
C     14   LPLOT         Plot unit number for linear solver information
C     15                 Reserved
C     16   NIWKFR OUT    First element of IWK which is free to be used
C                        as workspace between Newton iteration steps
C     17   NRWKFR OUT    First element of RWK which is free to be used
C                        as workspace between Newton iteration steps
C     18   LIWKA  OUT    Length of IWK actually required
C     19   LRWKA  OUT    Length of RWK actually required
C     20                 Reserved
C     21   LUGOUT IN     Logical unit number for summary plot 
C                        information
C     22                 Reserved
C     23   IFAIL  OUT    Set in case of failure of NJITSL (IERR=80),
C                        FCN (IERR=82) or JAC(IERR=83)
C                        to the nonzero IFAIL value returned by the 
C                        routine indicating the failure .
C     24..30             Reserved
C     31   NITMAX IN     Maximum number of permitted iteration
C                        steps (Default: 50)
C     32..40             Reserved
C     41   LITMAX IN     Maximum number of iterative linear solver 
C                        iteration steps to be done.
C     42..50             Reserved
C
C*   Optional REAL input/output in RWK:
C    ====================================
C
C     Pos. Name          Meaning
C
C      1..16             Reserved
C     17   CONV   OUT    The achieved relative accuracy after the  
C                        current step
C     18   SUMX   OUT    Natural level (named Normx  in printouts)
C                        of the current iterate, e.g. Norm2(DX)**2,
C                        with the Newton correction DX
C     19   DLEVF  OUT    Standard level (named Normf  in printouts)
C                        of the current iterate, e.g. Norm2(F(X))
C                        with the nonlinear model function F.
C     20   FCBND  IN     Bounded damping strategy restriction factor
C                        (Default is 10)
C     21   FCSTRT IN     Damping factor for first Newton iteration -
C                        overrides option NONLIN, if set (see note 5)
C     22   FCMIN  IN     Minimal allowed damping factor (see note 5)
C     23                 Reserved
C     24   SIGMA2 IN     Decision parameter about increasing damping
C                        factor to corrector if predictor is small.
C     25..31             Reserved
C     32   RHO    IN     Accuracy matching factor for the iterative
C                        linear solver. Must be .LE. 1/6.
C     33..35             Reserved
C     36..40             Reserved for usage by NXPLOT supplement
C     41   RHOORD IN     Good Broyden parameter:
C                        Security factor in error estimate (RHO.GE.1)
C                        in ordinary corrections computation.
C                        Default: 4.0D0
C     42   RHOSIM        Good Broyden parameter:
C                        Security factor in error estimate (RHO.GE.1)
C                        in simplified corrections computation.
C                        Default: 4.0D0
C     43   TAUMIN        Good Broyden parameter:
C                        Minimum accepted stepsize factor tk.
C                        Default  1.0D-8
C     44   TAUMAX        Good Broyden parameter:
C                        Maximum accepted stepsize factor tk.
C                        Default: 1.0D2
C     45   TAUEQU        Good Broyden parameter:
C                        Parameter for "near equal" cycle check of
C                        rejected tau values. Default: 1.0D-2
C     46..50             Reserved for iterative linear solver parameters
C
C     Note 5:
C       The default values of the internal parameters may be obtained
C       from the monitor output with at least IOPT field MPRMON set to 2
C       and by initializing the corresponding RWK-fields to zero.
C
C     Note 6:
C       Any output units which are used by this program (as selected
C       by option settings) need to be opened by a FORTRAN open-
C       statement before calling GIANT.  
C
C*   Error messages:
C    ===============
C
C      2    Termination after NITMAX iterations ( as indicated by
C           input parameter NITMAX )
C      3    Termination, since damping factor became to small
C           This error occurs due to (normally repetedly) failing
C           of a monotonicity test. This problem may be caused by
C           an insufficient accuracy of the output delivered by the
C           iterative linear solver. Possibly it will help in this
C           situation to increase RHOORD and perhaps RHOSIM (see
C           RWK(41) and RWK(42)) up to 40.0 or 400.0 . Another
C           possibility to get help may be to use an other
C           iterative linear solver.
C     10    Integer or real workspace too small
C     20    Bad input to dimensional parameter N
C     21    Nonpositive value for RTOL supplied
C     22    Negative scaling value via vector XSCAL supplied
C     30    One or more fields specified in IOPT are invalid
C           (for more information, see error-printout)
C     80    Error signalled by iterative linear solver routine ,
C           for more detailed information see IFAIL-value
C           stored to IWK(23)
C     82    Error signalled by user routine FCN (Nonzero value
C           returned via IFAIL-flag; stored to IWK(23) )
C     83    Error signalled by user routine JAC (Nonzero value
C           returned via IFAIL-flag; stored to IWK(23) )
C
C     Note 7 : in case of failure:
C        -    use better initial guess
C        -    or refine model
C        -    or apply continuation method to an appropriately
C             embedding of your problem function
C
C*    Machine dependent constants used:
C     =================================
C
C     DOUBLE PRECISION EPMACH  in  NJPCHK, NJINT
C     DOUBLE PRECISION GREAT   in  NJPCHK
C     DOUBLE PRECISION SMALL   in  NJPCHK, NJINT, NJSCAL
C
C*    Subroutines called: NJPCHK, NJINT
C
C     ------------------------------------------------------------
C*    End Prologue
C
C     Summary of changes:
C     -------------------
C
C     2.2    91, January    First release for CodeLib 
C     2.2.1  91, March      GBIT1I - replaced unscaled version by scaled
C                           version related subroutines changed: ITZMID,
C                           and SPRODI replaced by SPRODS.
C     2.2.2  91, August 28  Bounded damping strategy implemented
C     2.2.3  91, August 28  FCN-count changed for anal. Jacobian,
C                           RWK structured compatible to other 2.2.3
C                           version codes
C     2.3    91, Sept.  3   New release for CodeLib
C            00, July   12  RTOL output-value bug fixed
C
C     ------------------------------------------------------------
C
C     PARAMETER (IRWKI=xx, LRWKI=yy)  
C     IRWKI: Start position of internally used RWK part
C     LRWKI: Length of internally used RWK part
C     (actual values see parameter statement below)
C
C     INTEGER L5,L51,L6,L61,L76,L77,L8,L9,L14
C     Starting positions in RWK of formal array parameters of internal
C     routine N1INT (dynamically determined in driver routine GIANT,
C     dependent on N and options setting)
C
C     Further RWK positions (only internally used)
C
C     Position  Name     Meaning
C
C     IRWKI     FCKEEP   Damping factor of previous successfull iter.
C     IRWKI+1   FCA      Previous damping factor
C     IRWKI+2   FCPRI    A priori estimate of damping factor
C     IRWKI+3   DMYCOR   Number My of latest corrector damping factor
C                        (kept for use in rank-1 decision criterium)
C     IRWKI+4   HPOST    The estimate of the Cantorovitch constant H
C     IRWKI+5   EPSINK   A-posteriori tolerance of iterative linear
C                        solver solution
C     IRWKI+(6..LRWKI-1) Free
C
C     Internal arrays stored in RWK (see routine NJINT for descriptions)
C
C     Position  Array         Type   Remarks
C
C     L5        DX(N)         Perm
C     L51       DXQ(N)        Perm
C     L6        XA(N)         Perm
C     L61       F(N)          Perm
C     L76       DELX(N)       Perm
C     L77       XW(N)         Perm
C     L8                      low: Perm, high: Temp
C     L9        XWI(N)        Temp
C
      EXTERNAL NJINT,NJPCHK
      INTRINSIC DBLE
      INTEGER IRWKI, LRWKI
      PARAMETER (IRWKI=51, LRWKI=10)  
      DOUBLE PRECISION ONE
      PARAMETER (ONE=1.0D0)
      DOUBLE PRECISION TEN
      PARAMETER (TEN=1.0D1)
      DOUBLE PRECISION ZERO
      PARAMETER (ZERO=0.0D0)
      INTEGER NITMAX,LUERR,LUMON,LUSOL,MPRERR,MPRMON,MPRSOL,
     $NRWKFR,NRFRIN,NRW,NIWKFR,NIFRIN,NIW,NONLIN
      INTEGER L5,L51,L6,L61,L76,L77,L8,L9,L14,NRWLP,NIWLP,LITMAX,KMAX
      DOUBLE PRECISION FC,FCMIN,PERCI,PERCR
      LOGICAL QINIMO,QFCSTR,QSUCC,QBDAMP
      CHARACTER CHGDAT*20, PRODCT*8
C     Good Broyden internal parameters
      INTEGER KMAXDF,KMAXMI
      PARAMETER ( KMAXDF=9, KMAXMI=2 )
      DOUBLE PRECISION RHOORD,RHOSIM,TAUMIN,TAUMAX,TAUEQU
      PARAMETER( TAUMIN=1.0D-8, TAUMAX=1.0D2 , TAUEQU=1.0D-2 )
      PARAMETER (RHOORD=4.0D0,RHOSIM=4.0D0)
C     Which version ?
      LOGICAL QVCHK
      INTEGER IVER
      PARAMETER( IVER=142301 )
C
C     Version: 2.3             Latest change:
C     -----------------------------------------
C
      DATA      CHGDAT      /'July 12, 2000       '/
      DATA      PRODCT      /'GIANT   '/
C*    Begin
      IERR = 0
      QVCHK = IWK(12).LT.0
      IWK(12) = IVER
      IF (QVCHK) RETURN
C        Print error messages?
      MPRERR = IOPT(11)
      LUERR = IOPT(12)
      IF (LUERR .EQ. 0) THEN
        LUERR = 6
        IOPT(12)=LUERR
      ENDIF
C        Print iteration monitor?
      MPRMON = IOPT(13)
      LUMON = IOPT(14)
      IF (LUMON .LE. 0 .OR. LUMON .GT. 99) THEN
        LUMON = 6
        IOPT(14)=LUMON
      ENDIF
C        Print intermediate solutions?
      MPRSOL = IOPT(15)
      LUSOL = IOPT(16)
      IF (LUSOL .EQ. 0) THEN
        LUSOL = 6
        IOPT(16)=LUSOL
      ENDIF
C        Print linear solver monitor?
      MPRLIN = IOPT(17)
      LULIN = IOPT(18)
      IF (LULIN .EQ. 0) THEN
        LULIN = 6
        IOPT(18)=LULIN
      ENDIF
C        Print time summary statistics?
      MPRTIM = IOPT(19)
      LUTIM = IOPT(20)
      IF (LUTIM .EQ. 0) THEN
        LUTIM = 6
        IOPT(20)=LUTIM
      ENDIF
      QSUCC = IOPT(1).EQ.1
      QINIMO = MPRMON.GE.1.AND..NOT.QSUCC
C     Print GIANT heading lines
      IF(QINIMO)THEN
10000   FORMAT('   G I A N T *****  V e r s i o n  ',
     $         '2 . 3 ***',//,1X,'Generalized Newton-Methods ',
     $         'for the solution of nonlinear systems',//)
        WRITE(LUMON,10000)
      ENDIF
C     Check input parameters and options
      CALL NJPCHK(N,X,XSCAL,RTOL,IOPT,IERR,LIWK,IWK,LRWK,RWK)
C     Exit, if any parameter error was detected till here
      IF (IERR.NE.0) RETURN 
C
      LTYP=IOPT(8)
      IF (LTYP.EQ.0) LTYP=1
      IOPT(8)=LTYP
C     WorkSpace: RWK
      NRWLP=0
      L5=IRWKI+LRWKI
      L51=L5+N
      L6=L51+N
      L61=L6+N
      L76=L61+N
      L77=L76+N
      L8=L77+N
      NRWKFR = L8
      L14=LRWK+1
      L9=L14-N
C     End WorkSpace at NRW
C     WorkSpace: IWK
      NIWLP=N
      L23=51
      NIWKFR = L23
      NIW=L23-1
C     End WorkSpace at NIW
      IWK(16) = NIWKFR
      IWK(17) = NRWKFR
      NIWTMP=0
      NRWTMP=LRWK-L9+1
      NIW=NIWKFR-1+NIWTMP
      NRW=NRWKFR-1+NRWTMP
      LRWL=LRWK-NRW
      LIWL=LIWK-NIW
      IF (IOPT(41).EQ.0) IOPT(41)=KMAXDF
      IF (LTYP.EQ.1) THEN
        KMAX = IOPT(41)
        IF (IOPT(41).LT.-1) KMAX = 0
        IF (IOPT(41).EQ.-1) THEN
          KMAX = IDINT( DBLE(FLOAT(LRWK-NRW-4*N-17)) / DBLE(FLOAT(N+2)))
          KMAX = MAX0(KMAX,0)
          IF ( KMAX.LT.KMAXMI ) THEN
            IF (MPRERR.GE.1) WRITE (LUERR,10005) KMAX
10005       FORMAT(1X,'Workspace optimal KMAX would be ',I7,
     $                ' - but is too small')
            KMAX = KMAXMI
          ENDIF 
          IOPT(41) = KMAX
          IF (MPRMON.GE.1) WRITE (LUMON,10006) KMAX 
10006     FORMAT(1X,'selected KMAX is ',I7)
        ENDIF
        NRW = NRW + (N+2)*KMAX+4*N+17
      ENDIF
      NIFRIN = NIWKFR
      NRFRIN = NRWKFR
C
      IF(NRW.GT.LRWK.OR.NIW.GT.LIWK)THEN
        IERR=10
        NIFRIN = NIW+1
        NRFRIN = NRW+1
      ELSE
        IF(QINIMO)THEN
          PERCR = DBLE(NRW)/DBLE(LRWK)*100.0D0
          PERCI = DBLE(NIW)/DBLE(LIWK)*100.0D0
C         Print statistics concerning workspace usage
10050     FORMAT(' Real    Workspace declared as ',I9,
     $    ' is used up to ',I9,' (',F5.1,' percent)',//,
     $    ' Integer Workspace declared as ',I9,
     $    ' is used up to ',I9,' (',F5.1,' percent)',//)
          WRITE(LUMON,10050)LRWK,NRW,PERCR,LIWK,NIW,PERCI
        ENDIF
        IF(QINIMO)THEN
10051     FORMAT(/,' N =',I7,//,' Prescribed relative ',
     $    'precision',D10.2,/)
          WRITE(LUMON,10051)N,RTOL
10053     FORMAT(' The selected linear solver is :',/,' ',A)
          IF(LTYP.EQ.1) THEN
            WRITE(LUMON,10053) 'Good Broyden - ZIB CODELIB'
          ELSE IF(LTYP.EQ.9) THEN
            WRITE(LUMON,10053) 'A user supplied Solver'
          ENDIF
        ENDIF
        NONLIN=IOPT(31)
        IF (IOPT(38).EQ.0) QBDAMP = NONLIN.EQ.4
        IF (IOPT(38).EQ.1) QBDAMP = .TRUE.
        IF (IOPT(38).EQ.2) QBDAMP = .FALSE.
        IF (QBDAMP) THEN
          IF (RWK(20).LT.ONE) RWK(20) = TEN
        ENDIF
        IF (QINIMO) THEN
10065     FORMAT(' Problem is specified as being ',A)
          IF (NONLIN.EQ.1) THEN
            WRITE(LUMON,10065) 'linear'
          ELSE IF (NONLIN.EQ.2) THEN
            WRITE(LUMON,10065) 'mildly nonlinear'
          ELSE IF (NONLIN.EQ.3) THEN
            WRITE(LUMON,10065) 'highly nonlinear'
          ELSE IF (NONLIN.EQ.4) THEN
            WRITE(LUMON,10065) 'extremely nonlinear'
          ENDIF
10066     FORMAT(' Bounded damping strategy is ',A,:,/, 
     $           ' Bounding factor is ',D10.3)
          IF (QBDAMP) THEN
            WRITE(LUMON,10066) 'active', RWK(20)
          ELSE
            WRITE(LUMON,10066) 'off'
          ENDIF
        ENDIF
C       Maximum permitted number of iteration steps
        NITMAX=IWK(31)
        IF (NITMAX.LE.0) NITMAX=50
        IWK(31)=NITMAX
10078   FORMAT(' Maximum permitted number of iteration steps : ',
     $         I6)
        IF (QINIMO) WRITE(LUMON,10078) NITMAX
C       Initial damping factor for highly nonlinear problems
        QFCSTR=RWK(21).GT.ZERO
        IF (.NOT.QFCSTR) THEN
          RWK(21)=1.0D-2
          IF (NONLIN.EQ.4) RWK(21)=1.0D-4
        ENDIF
C       Minimal permitted damping factor
        IF (RWK(22).LE.ZERO) RWK(22)=1.0D-4
        FCMIN=RWK(22)
C       Decision parameter about increasing too small predictor
C       to greater corrector value
C       IF (RWK(24).LT.ONE) RWK(24)=10.0D0/FCMIN       
        IF (RWK(24).LT.ONE) RWK(24)=10.0D0     
C       Starting value of damping factor (FCMIN.LE.FC.LE.1.0)
        IF(NONLIN.LE.2.AND..NOT.QFCSTR)THEN
C         for linear or mildly nonlinear problems
          FC = ONE
        ELSE
C         for highly or extremely nonlinear problems
          FC = RWK(21)
        ENDIF
        RWK(21)=FC
        IF (MPRMON.GE.2.AND..NOT.QSUCC) THEN
10079     FORMAT(//,' Internal parameters:',//,
     $      ' Starting value for damping factor FCSTART = ',D9.2,/,
     $      ' Minimum allowed damping factor FCMIN = ',D9.2)
          WRITE(LUMON,10079) RWK(21),FCMIN
        ENDIF
        LITMAX = IWK(41)
        IF (LITMAX.LE.0) LITMAX=1000
        IWK(41) = LITMAX
C       Good Broyden parameter defaults
        IF (RWK(41).EQ.ZERO) RWK(41)=RHOORD
        IF (RWK(42).EQ.ZERO) RWK(42)=RHOSIM
        IF (RWK(43).EQ.ZERO) RWK(43)=TAUMIN
        IF (RWK(44).EQ.ZERO) RWK(44)=TAUMAX
        IF (RWK(45).EQ.ZERO) RWK(45)=TAUEQU
C       Store lengths of actually required workspaces
        IWK(18) = NIWKFR-1
        IWK(19) = NRWKFR-1
C
C     Initialize and start time measurements monitor
C
      IF ( IOPT(1).EQ.0 .AND. MPRTIM.NE.0 ) THEN
        CALL MONINI (' GIANT',LUTIM)
        CALL MONDEF (0,'GIANT')
        CALL MONDEF (1,'FCN')
        CALL MONDEF (2,'Jacobi')
        CALL MONDEF (3,'Precon')
        CALL MONDEF (4,'It-lin-sol')
        CALL MONDEF (5,'Muljac')
        CALL MONDEF (6,'Output')
        CALL MONSTR (IERR)
      ENDIF
C
C
        IERR=-1
C       If IERR is unmodified on exit, successive steps are required
C       to complete the Newton iteration
        CALL NJINT(N,FCN,JAC,X,XSCAL,RTOL,NITMAX,NONLIN,IOPT,IERR,
     $  LRWK,RWK,NRFRIN,LRWL,LIWK,IWK,NIFRIN,LIWL,
     $  LIWKU,IWKU,LRWKU,RWKU,MULJAC,PRECON,ITSOL,
     $  RWK(L5),RWK(L51),RWK(L6),RWK(L61),RWK(L77),RWK(L76),RWK(L9),
     $  RWK(21),RWK(22),RWK(24),RWK(IRWKI+1),RWK(IRWKI),RWK(IRWKI+2),
     $  RWK(IRWKI+3),RWK(17),RWK(18),RWK(19),RWK(IRWKI+4),
     $  RWK(IRWKI+5),MPRERR,MPRMON,MPRSOL,LUERR,
     $  LUMON,LUSOL,IWK(1),IWK(3),IWK(4),IWK(5),IWK(6),
     $  IWK(7),IWK(8),QBDAMP)
C
      IF (MPRTIM.NE.0.AND.IERR.NE.-1.AND.IERR.NE.10) THEN
          CALL MONHLT
          CALL MONPRT
        ENDIF
C
C       Free workspaces, so far not used between steps
        IWK(16) = NIWKFR
        IWK(17) = NRWKFR
C       Store lengths of actually required workspaces
        IF (IERR.GE.0) THEN
          IWK(18) = NIW + NIFRIN - NIWKFR
          IWK(19) = NRW + NRFRIN - NRWKFR
        ENDIF
      ENDIF
C     Print statistics
      IF (MPRMON.GE.1.AND.IERR.NE.-1.AND.IERR.NE.10) THEN
10080   FORMAT(/, '   ******  Statistics * ', A8, ' **********', /,
     $            '   ***  Newton iterations : ', I10,'  ***', /,
     $            '   ***  Corrector steps   : ', I10,'  ***', /,
     $            '   ***  JAC calls         : ', I10,'  ***', /,
     $            '   ***  FCN calls         : ', I10,'  ***', /,
     $            '   ***  Lin. Solv. (ord.) : ', I10,'  ***', /,
     $            '   ***  Lin. Solv. (sim.) : ', I10,'  ***', /,
     $            '   ***  Lin. Solv. (tot.) : ', I10,'  ***', /,
     $            '   ***  Lin. accur. fails : ', I10,'  ***', /,
     $            '   ****************************************', /)
        WRITE (LUMON,10080) PRODCT,IWK(1),IWK(3),IWK(5),
     $  IWK(4),IWK(6),IWK(7),IWK(6)+IWK(7),IWK(8)
      ENDIF
C     Print workspace requirements, if insufficient
      IF (IERR.EQ.10) THEN
10090   FORMAT(//,' ',20('*'),'Workspace Error',20('*'))
        IF (MPRERR.GE.1) WRITE(LUERR,10090)
        IF(NRW.GT.LRWK)THEN
10091     FORMAT(/,' Real Workspace dimensioned as',1X,I9,
     $    1X,'must be enlarged at least up to ',
     $    I9,//)
          IF (MPRERR.GE.1) WRITE(LUERR,10091)LRWK,NRFRIN-1
        ENDIF
        IF(NIW.GT.LIWK)THEN
10092     FORMAT(/,' Integer Workspace dimensioned as ',
     $    I9,' must be enlarged at least up ',
     $    'to ',I9,//)
          IF (MPRERR.GE.1) WRITE(LUERR,10092)LIWK,NIFRIN-1
        ENDIF
      ENDIF
      RETURN
      END
C
      SUBROUTINE NJPCHK(N,X,XSCAL,RTOL,IOPT,IERR,LIWK,IWK,LRWK,RWK)
C*    Begin Prologue NJPCHK
      INTEGER N
      DOUBLE PRECISION X(N),XSCAL(N)
      DOUBLE PRECISION RTOL
      INTEGER IOPT(50)
      INTEGER IERR
      INTEGER LIWK
      INTEGER IWK(LIWK)
      INTEGER LRWK
      DOUBLE PRECISION RWK(LRWK)
C     ------------------------------------------------------------
C
C*    Summary :
C
C     N J P C H K : Checking of input parameters and options
C                   for GIANT.
C
C*    Parameters:
C     ===========
C
C     See parameter description in driver routine.
C
C*    Subroutines called: ZIBCONST
C
C*    Machine dependent constants used:
C     =================================
C
C     EPMACH = relative machine precision
C     GREAT = squareroot of maxreal divided by 10
C     SMALL = squareroot of "smallest positive machine number
C             divided by relative machine precision"
      DOUBLE PRECISION EPMACH,GREAT,SMALL
C
C     ------------------------------------------------------------
C*    End Prologue
C
      INTRINSIC DBLE
      DOUBLE PRECISION ONE
      PARAMETER (ONE=1.0D0)
      DOUBLE PRECISION TEN
      PARAMETER (TEN=1.0D1)
      DOUBLE PRECISION ZERO
      PARAMETER (ZERO=0.0D0)
C
      PARAMETER (NUMOPT=50)
      INTEGER IOPTL(NUMOPT),IOPTU(NUMOPT)
      DOUBLE PRECISION TOLMIN,TOLMAX,DEFSCL
C
      DATA IOPTL /0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,-9999999,1,0,1,
     $            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
     $            0,0,0,0,0,-9999999,0,0,0,0,
     $            -9999999,-9999999,-9999999,-9999999,-9999999/
      DATA IOPTU /1,1,0,0,0,0,0,9,1,0,3,99,6,99,3,99,1,99,1,99,
     $            0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,
     $            0,0,2,0,0,9999999,0,0,0,0,
     $            9999999,9999999,9999999,9999999,9999999/
C
      CALL ZIBCONST(EPMACH,SMALL)
      GREAT  = TEN/SMALL
      IERR = 0
C        Print error messages?
      MPRERR = IOPT(11)
      LUERR = IOPT(12)
      IF (LUERR .LE. 0 .OR. LUERR .GT. 99) THEN
        LUERR = 6
        IOPT(12)=LUERR
      ENDIF
C
C     Checking dimensional parameter N
      IF ( N.LE.0 ) THEN
        IF (MPRERR.GE.1)  WRITE(LUERR,10011) N
10011   FORMAT(/,' Error: Bad input to dimensional parameter N supplied'
     $         ,/,8X,'choose N positive, your input is: N = ',I5)
        IERR = 20
      ENDIF
C
C     Problem type specification by user
      NONLIN=IOPT(31)
      IF (NONLIN.EQ.0) NONLIN=3
      IOPT(31)=NONLIN
C
C     Checking and conditional adaption of the user-prescribed RTOL
      IF (RTOL.LE.ZERO) THEN
        IF (MPRERR.GE.1) 
     $      WRITE(LUERR,'(/,A)') ' Error: Nonpositive RTOL supplied'
        IERR = 21
      ELSE
        TOLMIN = EPMACH*TEN*DBLE(N)
        IF(RTOL.LT.TOLMIN) THEN
          RTOL = TOLMIN
          IF (MPRERR.GE.2) 
     $      WRITE(LUERR,10012) 'increased ','smallest',RTOL
        ENDIF
        TOLMAX = 1.0D-1
        IF(RTOL.GT.TOLMAX) THEN
          RTOL = TOLMAX
          IF (MPRERR.GE.2) 
     $      WRITE(LUERR,10012) 'decreased ','largest',RTOL
        ENDIF
10012   FORMAT(/,' Warning: User prescribed RTOL ',A,'to ',
     $         'reasonable ',A,' value RTOL = ',D11.2)
      ENDIF
C     
C     Test user prescribed accuracy and scaling on proper values
      IF (N.LE.0) RETURN 
      IF (NONLIN.GE.3) THEN
        DEFSCL = RTOL
      ELSE
        DEFSCL = ONE
      ENDIF
      DO 10 I=1,N
        IF (XSCAL(I).LT.ZERO) THEN
          IF (MPRERR.GE.1) THEN 
            WRITE(LUERR,10013) I
10013       FORMAT(/,' Error: Negative value in XSCAL(',I5,') supplied')
          ENDIF
          IERR = 22
        ENDIF
        IF (XSCAL(I).EQ.ZERO) XSCAL(I) = DEFSCL
        IF ( XSCAL(I).GT.ZERO .AND. XSCAL(I).LT.SMALL ) THEN
          IF (MPRERR.GE.2) THEN
            WRITE(LUERR,10014) I,XSCAL(I),SMALL
10014       FORMAT(/,' Warning: XSCAL(',I5,') = ',D9.2,' too small, ',
     $             'increased to',D9.2)
          ENDIF
          XSCAL(I) = SMALL
        ENDIF
        IF (XSCAL(I).GT.GREAT) THEN
          IF (MPRERR.GE.2) THEN
            WRITE(LUERR,10015) I,XSCAL(I),GREAT
10015       FORMAT(/,' Warning: XSCAL(',I5,') = ',D9.2,' too big, ',
     $             'decreased to',D9.2)
          ENDIF
          XSCAL(I) = GREAT
        ENDIF
10    CONTINUE
C     Checks options
      DO 20 I=1,NUMOPT
        IF (IOPT(I).LT.IOPTL(I) .OR. IOPT(I).GT.IOPTU(I)) THEN
          IERR=30
          IF (MPRERR.GE.1) THEN
            WRITE(LUERR,20001) I,IOPT(I),IOPTL(I),IOPTU(I)
20001       FORMAT(' Invalid option specified: IOPT(',I2,')=',I12,';',
     $             /,3X,'range of permitted values is ',I8,' to ',I8)
          ENDIF
        ENDIF
20    CONTINUE
C     End of subroutine NJPCHK
      RETURN
      END
C
      SUBROUTINE NJINT(N,FCN,JAC,X,XSCAL,RTOL,NITMAX,NONLIN,IOPT,IERR,
     $LRWK,RWK,NRWKFR,LRWL,LIWK,IWK,NIWKFR,LIWL,LIWKU,IWKU,LRWKU,RWKU,
     $MULJAC,PRECON,ITSOL,
     $DX,DXQ,XA,F,XW,DELX,XWI,
     $FC,FCMIN,SIGMA2,FCA,FCKEEP,FCPRI,DMYCOR,
     $CONV,SUMX,DLEVF,HPOST,EPSINK,MPRERR,MPRMON,
     $MPRSOL,LUERR,LUMON,LUSOL,NITER,NCORR,NFCN,NJAC,
     $NLINOR,NLINSI,NAMCF,QBDAMP)
C*    Begin Prologue NJINT
      INTEGER N
      EXTERNAL FCN,JAC
      DOUBLE PRECISION X(N),XSCAL(N)
      DOUBLE PRECISION RTOL
      INTEGER NITMAX,NONLIN
      INTEGER IOPT(50)
      INTEGER IERR
      INTEGER LRWK
      DOUBLE PRECISION RWK(LRWK)
      INTEGER NRWKFR,LRWL,LIWK
      INTEGER IWK(LIWK)
      INTEGER NIWKFR,LIWL
      INTEGER LIWKU
      INTEGER IWKU(LIWKU)
      INTEGER LRWKU
      DOUBLE PRECISION RWKU(LRWKU)
      EXTERNAL MULJAC,PRECON,ITSOL
      DOUBLE PRECISION DX(N),DXQ(N),XA(N),F(N),XW(N),DELX(N),XWI(N)
      DOUBLE PRECISION FC,FCMIN,SIGMA2,FCA,FCKEEP,FCPRI,DMYCOR,CONV,
     $                 SUMX,DLEVF,HPOST,EPSINK
      INTEGER MPRERR,MPRMON,MPRSOL,LUERR,LUMON,LUSOL,NITER,
     $NCORR,NFCN,NJAC,NLINOR,NLINSI,NAMCF
      LOGICAL QBDAMP
C     ------------------------------------------------------------
C
C*    Summary :
C
C     N J I N T : Core routine for GIANT .
C     Damped Newton-algorithm for systems of highly nonlinear
C     equations especially designed for numerically sensitive
C     problems.
C
C*    Parameters:
C     ===========
C     (+ marks arrays to be kept over successive one-step-mode calls)
C
C       N,FCN,JAC,X(N),XSCAL(N),RTOL   
C                          See parameter description in driver routine
C
C       NITMAX      Int    Maximum number of allowed iterations
C       NONLIN      Int    Problem type specification
C                          (see IOPT-field NONLIN)
C       IOPT        Int    See parameter description in driver routine
C       IERR        Int    See parameter description in driver routine
C       LRWK        Int    Length of real workspace
C       RWK(LRWK)   Dble   Real workspace array
C       NRWKFR      Int    First free position of RWK on exit 
C       LRWL        Int    Holds the maximum amount of real workspace
C                          available to the linear solvers
C       LIWK        Int    Length of integer workspace
C       IWK(LIWK)   Int    Integer workspace array
C       NIWKFR      Int    First free position of IWK on exit 
C       LIWL        Int    Holds the maximum amount of integer workspace
C                          available to the linear solvers
C
C       LIWKU,IWKU(LIWKU),LRWKU,RWKU(LRWKU),MULJAC,PRECON,ITSOL
C                          See parameter description in driver routine
C
C    +  DX(N)       Dble   Actual Newton correction
C    +  DXQ(N)      Dble   Simplified Newton correction J(k-1)*X(k)
C    +  XA(N)       Dble   Previous Newton iterate
C    +  F(N)        Dble   Function (FCN) value of current iterate
C    +  XW(N)       Dble   Scaling factors for iteration vector
C    +  DELX(N)     Dble   For output from iterative linear solver:
C                          gets the difference vector between the 
C                          starting and the final iterate.
C       XWI(N)      Dble   Scaling values for iterative linear solver
C       FC          Dble   Actual Newton iteration damping factor.
C       FCMIN       Dble   Minimum permitted damping factor. If
C                          FC becomes smaller than this value, one
C                          of the following may occur:
C                          a.    Recomputation of the Jacobian
C                                matrix by means of difference
C                                approximation (instead of Rank1
C                                update), if Rank1 - update
C                                previously was used
C                          b.    Fail exit otherwise
C       SIGMA2      Dble   Decision parameter for damping factor
C                          increasing to corrector value
C       FCA         Dble   Previous Newton iteration damping factor.
C       FCKEEP      Dble   Keeps the damping factor as it is at start
C                          of iteration step.
C       FCPRI       Dble   Keeps the damping factor from the previous
C                          a-priori estimate
C       DMYCOR      Dble   Temporary value used during computation of 
C                          damping factors corrector.
C       CONV        Dble   Scaled maximum norm of the Newton-
C                          correction. Passed to RWK-field on output.
C       SUMX        Dble   Square of the natural level (see equal-
C                          named IOPT-output field)
C       DLEVF       Dble   Square of the standard level (see equal-
C                          named IOPT-output field)
C       HPOST       Dble   The estimate of the Cantorovitch constant H 
C       EPSINK      Dble   A-posteriori tolerance of iterative linear
C                          solvers solution
C       MPRERR,MPRMON,MPRSOL,LUERR,LUMON,LUSOL :
C                          See description of equal named IOPT-fields
C                          in the driver subroutine GIANT
C       NITER,NCORR,NFCN,NJAC,NLINOR,NLINSI :
C                          See description of equal named IWK-fields
C                          in the driver subroutine GIANT
C       QBDAMP      Logic  Flag, that indicates, whether bounded damping
C                          strategy is active:
C                          .true.  = bounded damping strategy is active
C                          .false. = normal damping strategy is active
C
C*    Internal double variables
C     =========================
C
C       CONVA    Holds the previous value of CONV .
C       DELNRM   Gets the (Euclidian) norm of the difference between
C                the starting and the final iterate of the iterative
C                linear solver
C       DIFNRM   Gets the (Euclidian) norm of the final correction
C                of the iterative linear solvers iterate.
C       DLEVFN   Standard level of the previous iterate.
C       DLEVXA   Natural level of the previous iterate.
C       DMYPRI   Temporary value used during computation of damping 
C                factors predictor.
C       DN       DBLE(FLOAT(N)) (unchanged)
C       EPSK     The a priori estimate for the iterative linear solvers
C                solution precision for the ordinary Newton correction.
C       EPSKK    The a priori estimate for the iterative linear solvers
C                solution precision for the simplified Newton 
C                correction. 
C       EPSIN    The precision input parameter to the iterative linear
C                solver 
C       FCDNM    Used to compute the denominator of the damping 
C                factor FC during computation of it's predictor,
C                corrector and aposteriori estimate (in the case of
C                performing a Rank1 update) .
C       FCH      Aposteriori estimate of FC for next iterate.
C       FCMIN2   FCMIN**2 . Used for FC-predictor computation.
C       FCNUMP   Gets the numerator of the predictor formula for FC.
C       FCNMP2   Temporary used for predictor numerator computation.
C       FCNUMK   Gets the numerator of the corrector computation 
C                of FC .
C       FCCOR    Gets the corrector estimate of the damping factor.
C       HPOST    Gets the Cantorovitch constant H as obtained during
C                the a posteriori estimate of the damping factor
C       RHO      Stopping criterium parameter for iterative
C                linear solver.
C       RSMALL   Additional upper threshold for the previous iterate
C                of the final iterate.
C       RTOLQ    RTOL*RTOL - used for convergence test SUMX<=RTOLQ*N .
C       SOLNRM   Gets the (Euclidian) norm of the solution obtained
C                by the iterative linear solver. 
C       SP       Temporary used during Broyden-step computations to
C                hold a scalar product.
C       SUMH     If the iterative solvers solution has to be compared
C                with the direct solvers one:
C                Gets the scaled norm of the difference between both
C                solutions.
C       SUMH2    If the iterative solvers solution has to be compared
C                with the direct solvers one: 
C                Gets the scaled norm of the direct solvers solution.
C       SUMXA    Square of natural level of the previous iterate.
C       S1       Temporary used for computation of norms
C       TH       Temporary variable used during corrector- and 
C                aposteriori computations of FC.
C
C*    Internal integer variables
C     ==========================
C
C     IFAIL      Gets the return value from subroutines called from
C                NJINT (FCN, JAC) 
C     ISCAL      Holds the scaling option from the IOPT-field ISCAL 
C     ITOPT(50)  The options array for the iterative solver (See
C                subroutine NJITSL for details).
C     K          Do loop index     
C     L1,L2      Do loop indices
C     LITMAX     Maximum number of iteration for iterative linear
C                solver (obtained from IWK(41))
C     MODE       Matrix storage mode (see IOPT-field MODE) 
C     NFCOUT     Gets the total number of nonlinear function 
C                evaluations (Jacobian evaluations appropriately 
C                included)
C     NRED       Count of successive corrector steps
C     NILUSE     Gets the amount of IWK used by the linear solver
C     NIWLA      Index of first element of IWK provided to the
C                linear solver
C     NRLUSE     Gets the amount of RWK used by the linear solver
C     NRWLA      Index of first element of RWK provided to the
C                linear solver
C
C
C*    Internal logical variables
C     ==========================
C
C     QINISC     Iterate initial-scaling flag:
C                =.TRUE.  : at first call of NJSCAL
C                =.FALSE. : at successive calls of NJSCAL
C     QSUCC      See description of IOPT-field QSUCC.
C     QLIINI     Initialization state of iterative linear solver 
C                workspace:
C                =.FALSE. : Not yet initialized
C                =.TRUE.  : Initialized - NJITSL has been called at
C                           least one time.
C     QNEXT      =.FALSE. : New Newton iterate not okay, one more
C                           damping factor reduction must be done.
C                =.TRUE.  : New Newton iterate okay, advance to next
C                           Newton iteration
C     QREP       =.FALSE. : Indicates, that the damping factor increase
C                           criterium didn't apply yet
C                =.TRUE.  : Indicates, that the damping factor increase
C                           criterium did apply one time just before.
C     QMIXIO     Indicates, that both to monitor and solution unit 
C                output will be done and that both units are the same.
C
C*    Internal constants:
C     ===================
C
C     none
C
C*    Subroutines called:
C     ===================
C
C      NJSOUT, NJPRV1, NJPRV2, NJSCAL,
C      NJITS1, NJITSL, ZIBCONST
C
C*    Machine constants used
C     ======================
C
      DOUBLE PRECISION EPMACH,SMALL
C 
C     ------------------------------------------------------------
C*    End Prologue
      EXTERNAL NJSOUT, NJPRV1, NJPRV2, NJSCAL,
     $         NJITS1, NJITSL
      INTRINSIC DSQRT,DMIN1,MAX0,MIN0
      DOUBLE PRECISION ZERO
      PARAMETER (ZERO=0.0D0)
      DOUBLE PRECISION ONE
      PARAMETER (ONE=1.0D0)
      DOUBLE PRECISION TWO
      PARAMETER (TWO=2.0D0)
      DOUBLE PRECISION HALF
      PARAMETER (HALF=0.5D0)
      DOUBLE PRECISION TEN
      PARAMETER (TEN=10.0D0)
      INTEGER IFIN
      PARAMETER (IFIN=2)
      INTEGER IFAIL,ISCAL,ITOPT(50),L1,LITMAX,MODE,MAXL1,MAXL1O,ITERM,
     $NFCOUT,NRED,NILUSE,NRLUSE,LULSOL,LUGOUT
      DOUBLE PRECISION CONVA,DELNRM,DIFNRM,DLEVFN,DLEVXA,DN,
     $DMYPRI,EPSK,EPSKK,EPSIN,FCBND,FCBH,FCDNM,FCH,FCMIN2,
     $FCNUMP,FCNMP2,FCNUMK,FCCOR,RHO,RSMALL,RTOLQ,SOLNRM,
     $SUMXA,S1,TH,APREC
      LOGICAL QINISC,QSUCC,QLIINI,QNEXT,QREP,QINCOR,QMIXIO
      CALL ZIBCONST(EPMACH,SMALL)
C*    Begin
C       ----------------------------------------------------------
C       1 Initialization
C       ----------------------------------------------------------
C       1.1 Control-flags and -integers
        QSUCC = IOPT(1).EQ.1
        ISCAL = IOPT(9)
        MODE = IOPT(2)
        LTYP = IOPT(8)
        QMIXIO = LUMON.EQ.LUSOL .AND. MPRMON.NE.0 .AND. MPRSOL.NE.0
        MPRTIM = IOPT(19)
        LUGOUT=IWK(21)
        IF (LUGOUT.LE.0 .OR. LUGOUT.GT.99) LUGOUT=0
C       Iterative solver's print level and unit
        ITOPT(13) = IOPT(17)
        ITOPT(14) = IOPT(18)
C       Pass time monitor on/off to linear solver
        ITOPT(19) = IOPT(19)
C
        ITOPT(22) = IWK(14)
        LULSOL = IWK(14)
C
        ITOPT(5) = LTYP
C       Special options for 'good Broyden'
        ITOPT(31)=IOPT(41)
        LITMAX=IWK(41)
C       ----------------------------------------------------------
C       1.3 Derivated internal parameters
        DN = DBLE(FLOAT(N))
        FCMIN2 = FCMIN*FCMIN
        RSMALL = TEN*RTOL
        RTOLQ = RTOL*RTOL
        RHO = RWK(32)
        IF (RHO.EQ.ZERO) RHO=1.0D0/6.0D0
C       ----------------------------------------------------------
C       1.4 Adaption of input parameters, if necessary
        IF(FC.LT.FCMIN) FC = FCMIN
        IF(FC.GT.ONE) FC = ONE
C       ----------------------------------------------------------
C       1.5 Initial preparations
        QLIINI = .FALSE.
        IFAIL = 0
        FCBND = ZERO
        IF (QBDAMP) FCBND = RWK(20)
        NRWLAI = NRWKFR
        NIWLAI = NIWKFR
        LRWLI = LRWL
        LIWLI = LIWL
        RWK(NRWLAI+1)=RWK(43)
        RWK(NRWLAI+2)=RWK(44)
        RWK(NRWLAI+3)=RWK(45)
C       ----------------------------------------------------------
C       1.5.1 Miscellaneous preparations of first iteration step
        IF (.NOT.QSUCC) THEN
          NITER = 0
          NCORR = 0
          NFCN = 0
          NJAC = 0
          NLINOR = 0
          NLINSI = 0
          NAMCF = 0
          ITOPT(48)=0
          ITOPT(49)=0
          QINISC = .TRUE.
          FCKEEP = FC
          FCA = FC
          FCPRI = FC
          HPOST = ONE/FC
          CONV = ZERO
          DO 1521 L1=1,N
            XA(L1)=X(L1)
1521      CONTINUE
C         ------------------------------------------------------
C         1.6 Print monitor header
          IF(MPRMON.GE.2 .AND. .NOT.QMIXIO)THEN
            IF (MPRTIM.NE.0) CALL MONON (6)
16003       FORMAT(///,2X,66('*'))
            WRITE(LUMON,16003)
16004       FORMAT(/,8X,'It',7X,'Normf ',10X,'Normx ',8X,'Damp.Fct.')
            WRITE(LUMON,16004)
            IF (MPRTIM.NE.0) CALL MONOFF(6)
          ENDIF
C         --------------------------------------------------------
C         1.7 Startup step
C         --------------------------------------------------------
C         1.7.1 Computation of the residual vector
          IF (MPRTIM.NE.0) CALL MONON (1)
          CALL FCN(N,X,F,RWKU,IWKU,NFCN,IFAIL)
          IF (MPRTIM.NE.0) CALL MONOFF(1)
          NFCN = NFCN+1
          DO 1522 L1=1,N
            DX(L1)=ZERO
            DXQ(L1)=ZERO
            DELX(L1)=ZERO
1522      CONTINUE
C     Exit, if ...
          IF (IFAIL.LT.0) THEN
            IERR = 82
            GOTO 4299
          ENDIF
        ELSE
          QINISC = .FALSE.
        ENDIF
C
C       Main iteration loop
C       ===================
C
C       Repeat
2       CONTINUE
C         --------------------------------------------------------
C         2 Startup of iteration step
C         ------------------------------------------------------
C         2.1 Scaling of variables X(N)
          CALL NJSCAL(N,X,XA,XSCAL,XW,ISCAL,QINISC,IOPT,LRWK,RWK)
          DO 21 L1=1,N
            XWI(L1)=XW(L1)
21        CONTINUE
          QINISC = .FALSE.
          IF(NITER.NE.0)THEN
C           ----------------------------------------------------
C           2.2 Aposteriori estimate of damping factor
            FCNUMP = ZERO
            DO 2201 L1=1,N
              FCNUMP=FCNUMP+(DX(L1)/XW(L1))**2
2201        CONTINUE
            FCDNM = ZERO
            DO 2202 L1=1,N
              FCDNM=FCDNM+(DELX(L1)/XW(L1))**2
2202        CONTINUE
            FCDNM = DMAX1(FCDNM,EPMACH*EPMACH)
            IF (FC.EQ.FCPRI) THEN
              FCA = FC
            ELSE
              DMYCOR = FCA*FCA*HALF*DSQRT(FCNUMP/FCDNM)
              HPOST = ONE/DMYCOR
              DMYCOR = DMYCOR*(ONE-EPSINK)
              IF (NONLIN.LE.3) THEN
                FCCOR = DMIN1(ONE,DMYCOR)
              ELSE
                FCCOR = DMIN1(ONE,HALF*DMYCOR)
              ENDIF
              FCA = DMAX1(DMIN1(FC,FCCOR),FCMIN)
C$Test-begin
              IF (MPRMON.GE.5) THEN
                WRITE(LUMON,22201) FCCOR, FC, DMYCOR, FCNUMP,
     $                             FCDNM
22201           FORMAT (/, ' +++ aposteriori estimate +++', /,
     $                  ' FCCOR  = ', D18.10, '  FC     = ', D18.10, /,
     $                  ' DMYCOR = ', D18.10, '  FCNUMP = ', D18.10, /,
     $                  ' FCDNM  = ', D18.10, /,
     $                  ' ++++++++++++++++++++++++++++', /)
              ENDIF
C$Test-end 
            ENDIF
C           ------------------------------------------------------
C           2.2.1 Computation of the numerator of damping
C                 factor predictor
            FCNMP2 = ZERO
            DO 221 L1=1,N
              FCNMP2=FCNMP2+(DXQ(L1)/XW(L1))**2
221         CONTINUE
            FCNMP2 = DMAX1(FCNMP2,EPMACH*EPMACH)
            FCNUMP = FCNUMP*FCNMP2
          ENDIF
C         --------------------------------------------------------
C         2.3 Jacobian matrix (stored to array ASAVE(M2,N))
C         --------------------------------------------------------
C         2.3.1 Jacobian generation by routine JAC
          IF (MPRTIM.NE.0) CALL MONON (2)
          CALL JAC(FCN,N,X,XW,F,RWKU,IWKU,NJAC,IFAIL)
          IF (MPRTIM.NE.0) CALL MONOFF(2)
          NJAC = NJAC + 1
C     Exit, If ...
          IF (IFAIL.LT.0) THEN
            IERR = 83
            GOTO 4299
          ENDIF
C         --------------------------------------------------------
C         3 Central part of the Newton iteration step
C         --------------------------------------------------------
C         3.1 Iterative solution of the linear (N,N)-system
          CALL NJITS1(N,LITMAX,EPSK,XWI,MULJAC,PRECON,ITSOL,
     $               DX,DELX,F,ITOPT,
     $               LRWLI,RWK,NRWLAI,NRWKFR,
     $               LIWLI,IWK,NIWLAI,NIWKFR,
     $               LIWKU,IWKU,LRWKU,RWKU,
     $               DELNRM,SOLNRM,FCNUMP,RHO,XW,MAXL1O,ITERM,
     $               NITER,DXQ,HPOST,FC,FCA,FCMIN,FCMIN2,EPSINK,
     $               RTOL,QLIINI,LTYP,MPRMON,LUMON,MPRTIM,DLEVXA,
     $               NLINOR,NAMCF,IFIN,NONLIN,IFAIL)
          IF(IFAIL.NE.0) THEN
            IERR = 80
            GOTO 4299
          ENDIF
          DO 31 J=1,N
            DX(J) = -DX(J)
31        CONTINUE
C         --------------------------------------------------------
C         3.2 Evaluation of scaled natural level function SUMX
C             scaled maximum error norm CONV
C             evaluation of (scaled) standard level function
C             DLEVF ( DLEVF only, if MPRMON.GE.2 )
C             and computation of ordinary Newton corrections 
C             DX(N)
          SUMX=SOLNRM**2
          CONV=ZERO
          DLEVF=ZERO
          DO 320 J=1,N
            S1=DABS(DX(J)/XW(J))
            IF (S1.GT.CONV) CONV=S1
            DLEVF=DLEVF+F(J)*F(J)
320       CONTINUE
          DLEVF = DSQRT( DLEVF/DBLE(FLOAT(N)) )
C         --------------------------------------------------------
C         3.2.2 Save previous values
          DO 321 L1=1,N
            XA(L1)=X(L1)
321       CONTINUE
          SUMXA = SUMX
          DLEVXA = DSQRT(SUMXA/DBLE(FLOAT(N)))
          CONVA = CONV
C         --------------------------------------------------------
C         3.3 A - priori estimate of damping factor FC
          IF(NITER.NE.0.AND.NONLIN.NE.1)THEN
            FCDNM = (DELNRM*SOLNRM)**2
            FCNUMP=FCNUMP*(ONE-EPSINK)*(ONE-EPSINK)
            IF(FCDNM.GT.FCNUMP*FCMIN2 .OR.
     $        (NONLIN.EQ.4 .AND. FCA**2*FCNUMP .LT. 4.0D0*FCDNM)) THEN
              DMYPRI = FCA*DSQRT(FCNUMP/FCDNM)
            ELSE
              DMYPRI = ONE/FCMIN
            ENDIF
            HPOST = (ONE-EPSINK) / DMYPRI
            FCPRI = DMIN1(DMYPRI,ONE)
            IF (NONLIN.EQ.4) FCPRI = DMIN1(HALF*DMYPRI,ONE)
C$Test-begin
            IF (MPRMON.GE.5) THEN
              WRITE(LUMON,33201) FCPRI, FC, FCA, DMYPRI, FCNUMP,
     $                           FCDNM
33201         FORMAT (/, ' +++ apriori estimate +++', /,
     $                ' FCPRI  = ', D18.10, '  FC     = ', D18.10, /,
     $                ' FCA    = ', D18.10, '  DMYPRI = ', D18.10, /,
     $                ' FCNUMP = ', D18.10, '  FCDNM  = ', D18.10, /,
     $                   ' ++++++++++++++++++++++++', /)
            ENDIF
C$Test-end 
            FC = DMAX1(FCPRI,FCMIN)
            IF (QBDAMP) THEN
              FCBH = FCA*FCBND
              IF (FC.GT.FCBH) THEN
                FC = FCBH
                IF (MPRMON.GE.4)
     $            WRITE(LUMON,*) ' *** incr. rest. act. (a prio) ***'
              ENDIF
              FCBH = FCA/FCBND
              IF (FC.LT.FCBH) THEN
                FC = FCBH
                IF (MPRMON.GE.4)
     $            WRITE(LUMON,*) ' *** decr. rest. act. (a prio) ***'
              ENDIF
            ENDIF
          ENDIF
C         --------------------------------------------------------
C         3.4 Save natural level for later computations of
C             corrector and print iterate
          FCNUMK = SUMX
          IF (MPRMON.GE.2) THEN
            IF (MPRTIM.NE.0) CALL MONON (6)
            CALL NJPRV1(DLEVF,DLEVXA,FCKEEP,NITER,MPRMON,LUMON,
     $                  QMIXIO)
            IF (MPRTIM.NE.0) CALL MONOFF(6)
          ENDIF
          NRED = 0
          QNEXT = .FALSE.
C         QREP = .FALSE.            (always)  - or  
C         QREP = NITER .GT. 0       (first iterate only)
C         QREP = .TRUE.             (never)
          QREP = .TRUE.
C
C         Damping-factor reduction loop
C         ================================
C         DO (Until)
34        CONTINUE
            QINCOR = .FALSE.
C           ------------------------------------------------------
C           3.5 Preliminary new iterate
            DO 35 L1=1,N
              X(L1)=XA(L1)+DX(L1)*FC
35          CONTINUE
C           -----------------------------------------------------
C           3.5.2 Exit, if problem is specified as being linear
C     Exit Repeat If ...
            IF( NONLIN.EQ.1 )THEN
              IERR = 0
              GOTO 4299
            ENDIF
C           
C           ------------------------------------------------------
C           3.6.1 Computation of the residual vector
            IF (MPRTIM.NE.0) CALL MONON (1)
            CALL FCN(N,X,F,RWKU,IWKU,NFCN,IFAIL)
            IF (MPRTIM.NE.0) CALL MONOFF(1)
            NFCN = NFCN+1
C     Exit, If ...
            IF(IFAIL.LT.0)THEN
              IERR = 82
              GOTO 4299
            ENDIF
C             --------------------------------------------------
C             3.6.3 Solution of linear (N,N)-system
            MAXL1=LITMAX
            TH = FC-ONE
            DO 3641 L1=1,N
              DXQ(L1) = TH * DX(L1)
              DELX(L1) = ZERO
3641        CONTINUE
            RWK(NRWLAI)=RWK(42)
            EPSIN=RHO/(ONE+TWO*RHO)
            ITOPT(33)=IFIN
            ITOPT(1)=0
            IF (MPRMON.GE.4) THEN
              WRITE(LUMON,36100) EPSIN
36100         FORMAT(' +++ EPS := ',D10.3,' +++',/,
     $               ' Starting simplified Newton-correction comp.')
            ENDIF
            IF (LULSOL.GT.0) WRITE(LULSOL,36102) NITER,NRED+1
36102       FORMAT(' Simplified correction; niter=',I3,'; no=',I1)
            EPSKK = EPSIN
            IF (MPRTIM.NE.0) CALL MONON (4)
            CALL NJITSL(N,MAXL1,EPSIN,XWI,MULJAC,PRECON,
     $                 ITSOL,DXQ,DELX,F,ITOPT,
     $                 LRWLI,RWK(NRWLAI),NRLUSE,
     $                 LIWLI,IWK(NIWLAI),NILUSE,
     $                 LIWKU,IWKU,LRWKU,RWKU,
     $                 DELNRM,SOLNRM,DIFNRM,IFAIL)
            IF (MPRTIM.NE.0) CALL MONOFF(4)
            NLINSI=NLINSI+MAXL1
            IF (MPRMON.GE.4)
     $        WRITE(LUMON,36101) MAXL1,EPSIN
36101         FORMAT(' +++ Simpl. Newton correcture:  +++',/,2X,
     $              '#Iter = ',I4,2X,'Est. Rel. prec. = ',D10.3,/,
     $              ' ++++++++++')
            DELNRM = ZERO
            SOLNRM = ZERO
            DO 36118 J=1,N
              DXQ(J) = -DXQ(J)
              DELNRM = DELNRM + (DELX(J)/XW(J))**2
              SOLNRM = SOLNRM + (DXQ(J)/XW(J))**2
36118       CONTINUE
            DELNRM = DSQRT(DELNRM)
            SOLNRM = DSQRT(SOLNRM)
            IF(IFAIL.LT.0) THEN
              IERR = 80
              GOTO 4299
            ENDIF
            CONV=ZERO
            DLEVFN=ZERO
            DO 3642 J=1,N
              S1=DABS(DXQ(J)/XW(J))
              IF (S1.GT.CONV) CONV=S1
              DLEVFN=DLEVFN+F(J)*F(J)
3642        CONTINUE
            DLEVFN = DSQRT( DLEVFN/DN )
            SUMX=SOLNRM**2
            CALL NJGOUT(LUGOUT,NITER,NRED,DSQRT(SUMXA/DN),
     $                  DSQRT(SUMX/DN),FC,DBLE(FLOAT(MAXL1O)),
     $                  DBLE(FLOAT(MAXL1)),EPSK,DBLE(FLOAT(ITERM)))
C             -----------------------------------------------------
C             3.6.5 Convergence test
C     Exit Repeat If ...
            IF( SUMX.LE.RTOLQ*DN .AND. SUMXA.LE.RSMALL*DN )THEN
              IERR = 0
              GOTO 4299
            ENDIF
C           
            FCA = FC
C             ----------------------------------------------------
C             3.6.6 Evaluation of reduced damping factor
            DELNRM = DMAX1(DELNRM,EPMACH)
            HPOST=DELNRM/(FCA*FCA*HALF*DSQRT(FCNUMK))
            EPSINK=RHO*DMIN1(ONE/(ONE+RHO),HPOST) 
            DMYCOR=ONE/HPOST
            DMYCOR = DMYCOR*(ONE-EPSINK)
            IF (NONLIN.LE.3) THEN
              FCCOR = DMIN1(ONE,DMYCOR)
            ELSE
              FCCOR = DMIN1(ONE,HALF*DMYCOR)
            ENDIF
C$Test-begin
            IF (MPRMON.GE.5) THEN
              WRITE(LUMON,39001) FCCOR, FC, DMYCOR, FCNUMK,
     $                           FCDNM, FCA
39001           FORMAT (/, ' +++ corrector computation +++', /,
     $              ' FCCOR  = ', D18.10, '  FC     = ', D18.10, /,
     $              ' DMYCOR = ', D18.10, '  FCNUMK = ', D18.10, /,
     $              ' FCDNM  = ', D18.10, '  FCA    = ', D18.10, /,
     $                 ' +++++++++++++++++++++++++++++', /)
            ENDIF
C$Test-end
            IF (.NOT.QREP .AND. FCCOR.GT.SIGMA2*FCA) THEN
              IF(MPRMON.GE.3) THEN
                IF (MPRTIM.NE.0) CALL MONON (6)
                CALL NJPRV2(DLEVFN,DSQRT(SUMX/DBLE(FLOAT(N))),FC,
     $                      NITER,MPRMON,LUMON,QMIXIO,'+')
                IF (MPRTIM.NE.0) CALL MONOFF(6)
              ENDIF
              FC = FCCOR
C$Test-begin
              IF (MPRMON.GE.5) THEN
                WRITE(LUMON,39003) FC
39003             FORMAT (/, ' +++ corrector setting 2 +++', /,
     $                  ' FC     = ', D18.10, /,
     $                     ' +++++++++++++++++++++++++++', /)
              ENDIF
C$Test-end 
              QREP = .TRUE.
              QINCOR = .TRUE.
            ENDIF
C           ------------------------------------------------------
C           3.7 Natural monotonicity test
            IF(SUMX*(ONE-TWO*EPSKK).GT.SUMXA*(ONE+TWO*EPSK) .AND. 
     $         .NOT. QINCOR )THEN
C             ----------------------------------------------------
C             3.8 Output of iterate
              IF(MPRMON.GE.3) THEN
                IF (MPRTIM.NE.0) CALL MONON (6)
                CALL NJPRV2(DLEVFN,DSQRT(SUMX/DBLE(FLOAT(N))),FC,
     $                      NITER,MPRMON,LUMON,QMIXIO,'*')
                IF (MPRTIM.NE.0) CALL MONOFF(6)
              ENDIF
              FCH = DMIN1(FCCOR,HALF*FC)
              IF (FC.GT.FCMIN) THEN
                FC=DMAX1(FCH,FCMIN)
              ELSE
                FC=FCH
              ENDIF
              IF (QBDAMP) THEN
                FCBH = FCA/FCBND
                IF (FC.LT.FCBH) THEN
                  FC = FCBH
                  IF (MPRMON.GE.4)
     $              WRITE(LUMON,*) ' *** decr. rest. act. (a post) ***'
                ENDIF
              ENDIF
C$Test-begin
              IF (MPRMON.GE.5) THEN
                WRITE(LUMON,39002) FC
39002           FORMAT (/, ' +++ corrector setting 1 +++', /,
     $                  ' FC     = ', D18.10, /,
     $                     ' +++++++++++++++++++++++++++', /)
              ENDIF
C$Test-end 
              QREP = .TRUE.
              NCORR = NCORR+1
              NRED = NRED+1
C             ----------------------------------------------------
C             3.10 If damping factor is too small: fail exit
C     Exit Repeat If ...
              IF(FC.LT.FCMIN.AND.NRED.GT.1)THEN
                IERR = 3
                GOTO 4299
              ENDIF
            ELSE
              QNEXT = .NOT. QINCOR
            ENDIF
          IF(.NOT.(QNEXT)) GOTO  34
C         UNTIL ( expression - negated above)
C         End of damping-factor reduction loop
C         =======================================
C         ------------------------------------------------------
C         4 Preparations to start the following iteration step
C         ------------------------------------------------------
C         4.1 Print values
          IF(MPRMON.GE.3) THEN
            IF (MPRTIM.NE.0) CALL MONON (6)
            CALL NJPRV2(DLEVFN,DSQRT(SUMX/DBLE(FLOAT(N))),FC,NITER+1,
     $                  MPRMON,LUMON,QMIXIO,'*')
            IF (MPRTIM.NE.0) CALL MONOFF(6)
          ENDIF
C         Print the natural level of the current iterate and return
C         it in one-step mode
          SUMX = SUMXA
          IF(MPRSOL.GE.2.AND.NITER.NE.0) THEN
            IF (MPRTIM.NE.0) CALL MONON (6)
            CALL NJSOUT(N,XA,2,IOPT,RWK,LRWK,IWK,LIWK,RWKU,IWKU,
     $                  MPRSOL,LUSOL)
            IF (MPRTIM.NE.0) CALL MONOFF(6)
          ELSE IF(MPRSOL.GE.1.AND.NITER.EQ.0)THEN
            IF (MPRTIM.NE.0) CALL MONON (6)
            CALL NJSOUT(N,XA,1,IOPT,RWK,LRWK,IWK,LIWK,RWKU,IWKU,
     $                  MPRSOL,LUSOL)
            IF (MPRTIM.NE.0) CALL MONOFF(6)
          ENDIF
          NITER = NITER+1
          DLEVF = DLEVFN
C     Exit Repeat If ...
          IF(NITER.GE.NITMAX)THEN
            IERR = 2
            GOTO 4299
          ENDIF
          FCKEEP = FC
C         ------------------------------------------------------
C         4.2 Return, if in one-step mode
C
C Exit Subroutine If ...
          IF (MODE.EQ.1) THEN
            IWK(9)=ITOPT(43)
            IWK(18)=NIWLAI-1
            IWK(19)=NRWLAI-1
            IOPT(1)=1
            RETURN
          ENDIF
4298      CONTINUE
        GOTO 2
C       End Repeat
4299    CONTINUE
C       End of main iteration loop
C       ==========================
C       ----------------------------------------------------------
C       9 Exits
C       ----------------------------------------------------------
C       9.1 Solution exit
        APREC = -1.0D0
C 
        IF(IERR.EQ.0)THEN
          IF (NONLIN.NE.1) THEN
            APREC = DSQRT(SUMX/DBLE(FLOAT(N)))
C           Print final monitor output
            IF(MPRMON.GE.2) THEN
              IF (MPRTIM.NE.0) CALL MONON (6)
              CALL NJPRV2(DLEVFN,DSQRT(SUMX/DBLE(FLOAT(N))),FC,NITER+1,
     $                    MPRMON,LUMON,QMIXIO,'*')
              IF (MPRTIM.NE.0) CALL MONOFF(6)
            ENDIF
            IF(MPRMON.GE.1) THEN
              NFCOUT = NFCN
91001         FORMAT(///' Solution of nonlinear system ',
     $        'of equations obtained',/,' GIANT required',I3,
     $        ' Iteration steps with',I4,' Function ',
     $        'evaluations',//,' Achieved relative accuracy',D10.3)
                WRITE(LUMON,91001)NITER+1,NFCOUT,APREC
            ENDIF
          ELSE
            APREC = EPSINK/(ONE+EPSINK)
            IF(MPRMON.GE.1) THEN
91002         FORMAT(///' Solution of linear system ',
     $        'of equations obtained by GIANT',//,
     $        ' Achieved relative accuracy',D10.3
     $        ,2X)
                WRITE(LUMON,91002) APREC
            ENDIF
          ENDIF
        ENDIF
C       ----------------------------------------------------------
C       9.2 Fail exit messages
C       ----------------------------------------------------------
C       9.2.2 Termination after more than NITMAX iterations
        IF(IERR.EQ.2.AND.MPRERR.GE.1)THEN
92201     FORMAT(/,' Iteration terminates after NITMAX ',
     $    '=',I3,'  Iteration steps')
          WRITE(LUERR,92201)NITMAX
        ENDIF
C       ----------------------------------------------------------
C       9.2.3 Damping factor FC became too small
        IF(IERR.EQ.3.AND.MPRERR.GE.1)THEN
92301     FORMAT(/,' Damping factor has become too ',
     $    'small: lambda =',D10.3,2X,/,
     $    ' for more information, see errormessages list',/)
          WRITE(LUERR,92301)FC
        ENDIF
C       ----------------------------------------------------------
C       9.2.5 Error exit due to linear solver routine NJITSL
        IF(IERR.EQ.80.AND.MPRERR.GE.1)THEN
92501     FORMAT(/,' Error ',I5,' signalled by linear solver NJITSL')
          WRITE(LUERR,92501) IFAIL
        ENDIF
C       ----------------------------------------------------------
C       9.2.7 Error exit due to fail of user function FCN
        IF(IERR.EQ.82.AND.MPRERR.GE.1)THEN
92701     FORMAT(/,' Error ',I5,' signalled by user function FCN')
          WRITE(LUERR,92701) IFAIL
        ENDIF
C       ----------------------------------------------------------
C       9.2.7 Error exit due to fail of user function JAC
        IF(IERR.EQ.83.AND.MPRERR.GE.1)THEN
92801     FORMAT(/,' Error ',I5,' signalled by user function JAC')
          WRITE(LUERR,92801) IFAIL
        ENDIF
        IF(IERR.GE.80.AND.IERR.LE.83) IWK(23) = IFAIL
        IF ((IERR.EQ.82.OR.IERR.EQ.83).AND.NITER.LE.1.AND.MPRERR.GE.1)
     $  THEN
          WRITE (LUERR,92810)
92810     FORMAT(' Try to find a better initial guess for the solution')
        ENDIF
C       ----------------------------------------------------------
C       9.3 Common exit
        IF (MPRERR.GE.3.AND.IERR.NE.0.AND.NONLIN.NE.1) THEN
92102     FORMAT(/,'    Achieved relative accuracy',D10.3,2X)
          WRITE(LUERR,92102)CONVA
          APREC = CONVA
        ENDIF
        RTOL = APREC
        SUMX = SUMXA
        IF(MPRSOL.GE.2.AND.NITER.NE.0) THEN
           MODE=2
           IF (MPRTIM.NE.0) CALL MONON (6)
           CALL NJSOUT(N,XA,MODE,IOPT,RWK,LRWK,IWK,LIWK,RWKU,IWKU,
     $                 MPRSOL,LUSOL)
           IF (MPRTIM.NE.0) CALL MONOFF(6)
        ELSE IF(MPRSOL.GE.1.AND.NITER.EQ.0)THEN
           IF (MPRTIM.NE.0) CALL MONON (6)
           CALL NJSOUT(N,XA,1,IOPT,RWK,LRWK,IWK,LIWK,RWKU,IWKU,
     $                 MPRSOL,LUSOL)
           IF (MPRTIM.NE.0) CALL MONOFF(6)
        ENDIF
        NITER = NITER+1
        DLEVF = DLEVFN
        IWK(9) = ITOPT(43)
        IWK(10) = ITOPT(48)
        IWK(11) = ITOPT(49)
        IF(MPRSOL.GE.1)THEN
C         Print Solution or final iteration vector
          IF(IERR.EQ.0)THEN
             MODEFI = 3
          ELSE
             MODEFI = 4
          ENDIF
          IF (MPRTIM.NE.0) CALL MONON (6)
          CALL NJSOUT(N,X,MODEFI,IOPT,RWK,LRWK,IWK,LIWK,RWKU,IWKU,
     $                MPRSOL,LUSOL)
          IF (MPRTIM.NE.0) CALL MONOFF(6)
        ENDIF
C       Return the latest internal scaling to XSCAL
        DO 93 I=1,N
          XSCAL(I)=XW(I)
93      CONTINUE
C       End of exits
C       End of subroutine GIANT
      RETURN
      END
C
      SUBROUTINE NJITS1(N,LITMAX,EPSK,XWI,MULJAC,PRECON,ITSOL,
     $                  DX,DELX,F,ITOPT,
     $                  LRWLI,RWK,NRWLAI,NRWKFR,
     $                  LIWLI,IWK,NIWLAI,NIWKFR,
     $                  LIWKU,IWKU,LRWKU,RWKU,
     $                  DELNRM,SOLNRM,FCNUMP,RHO,XW,MAXL1,ITERM,
     $                  NITER,DXQ,HPOST,FC,FCA,FCMIN,FCMIN2,EPSINK,
     $                  RTOL,QLIINI,LTYP,MPRMON,LUMON,MPRTIM,DLEVXA,
     $                  NLINOR,NAMCF,IFIN,NONLIN,IFAIL)
C*    Begin Prologue NJITS1
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INTEGER N,LITMAX
      DOUBLE PRECISION EPSK,XWI(N)
      EXTERNAL MULJAC,PRECON,ITSOL
      DOUBLE PRECISION DX(N),DELX(N),F(N)
      INTEGER ITOPT(50)
      INTEGER LRWLI
      DOUBLE PRECISION RWK(*)
      INTEGER NRWLAI,NRWKFR,LIWLI
      INTEGER IWK(*)
      INTEGER NIWLAI,NIWKFR
      INTEGER LIWKU
      INTEGER IWKU(LIWKU)
      INTEGER LRWKU
      DOUBLE PRECISION RWKU(LRWKU)
      DOUBLE PRECISION DELNRM,SOLNRM,FCNUMP,RHO
      DOUBLE PRECISION XW(N),DXQ(N),HPOST,FC,FCA,FCMIN,FCMIN2,RTOL,
     $                 DLEVXA,EPSINK
      INTEGER MAXL1,ITERM,NITER,LTYP,MPRMON,LUMON,MPRTIM,NRLUSE,
     $        NILUSE,NLINOR,NAMCF,IFIN,NONLIN,IFAIL
      LOGICAL QLIINI
C     ------------------------------------------------------------
C
C*    Summary :
C
C     N J I T S 1 - Iterative solution of a linear system (implicitly
C                   given matrix by subroutine MULJAC -
C                   computes A*vector), with various precision
C                   control strategies and a-posteriori iteration.
C                   Calls the iterative solver via subroutine NJITSL.
C
C*    Parameters :
C     ============
C    
C     N           Int    Number of variables and nonlinear equations
C     LITMAX      Int    See MAXIT of subroutine NJITSL
C     EPSK        Dble   General usage: see subroutine NJITSL
C                        on output: finally required accuracy, passed
C                        for usage by monotonicity test.
C     XWI(N)      Dble   Scaling vector for iterative linear solver
C     MULJAC      Ext    See driver subroutine GIANT
C     PRECON      Ext    See driver subroutine GIANT
C     ITSOL       Ext    See driver subroutine GIANT
C     DX(N)       Dble   See SOL(N) of subroutine NJITSL
C     DELX(N)     Dble   See DEL(N) of subroutine NJITSL
C     F(N)        Dble   See RHS(N) of subroutine NJITSL
C     ITOPT(50)   Int    Iterative linear solvers option array
C     LRWLI       Int    Length of real workspace part for iterative
C                        linear solver
C     RWK(*)      Dble   The real workspace passed as parameter RWK
C                        to the driver subroutine GIANT
C     NRWLAI      Int    Starting position of the iterative linear
C                        solvers real workspace in array RWK(*)
C     NRWKFR      Int    First free position in real workspace RWK(*)
C                        (updated on first call of linear solver)
C     LIWLI       Int    Length of integer workspace part for iterative
C                        linear solver
C     IWK(*)      Int    The integer workspace passed as parameter IWK
C                        to the driver subroutine GIANT
C     NIWLAI      Int    Starting position of the iterative linear
C                        solvers integer workspace in array IWK(*)
C     NIWKFR      Int    First free position in integer workspace IWK(*)
C                        (updated on first call of linear solver)
C     LIWKU       Int    See subroutine NJITSL
C     IWKU(LIWKU) Int    See subroutine NJITSL
C     LRWKU       Int    See subroutine NJITSL
C     RWKU(LRWKU) Dble   See subroutine NJITSL
C     DELNRM      Dble   The Norm of DELX.
C                        Recomputed by this subroutine to preserve
C                        consistency in norm usage in GIANT.
C     SOLNRM      Dble   The Norm of DX.
C                        Recomputed by this subroutine to preserve
C                        consistency in norm usage in GIANT.
C     FCNUMP      Dble   See internal variables description in NJINT
C     RHO         Dble   See RWK(32) description in driver subroutine
C                        GIANT
C     XW(N)       Dble   See parameters description of NJINT
C     DXQ(N)      Dble   See parameters description of NJINT
C     HPOST       Dble   See RWK(IRWKI+4) description in driver
C                        subroutine GIANT 
C     FC          Dble   See parameters description of NJINT
C     FCA         Dble   See parameters description of NJINT
C     RTOL        Dble   See driver subroutine GIANT
C     DLEVXA      Dble   See internal variables description in NJINT
C     EPSINK      Dble   A-posteriori tolerance of iterative linear
C                        solvers solution
C                        on output: EPSK(achieved accuracy), passed
C                        for computation of a priori damping factor.
C     MAXL1       Int    Out: Number of linear solver iterations done 
C     ITERM       Int    Out: Indicator, which part of combined 
C                        termination criterium led to termination
C     NITER       Int    See IWK(1) as described in driver GIANT
C     LTYP        Int    See IOPT(8) as described in driver GIANT
C     MPRMON      Int    See IOPT(13) as described in driver GIANT
C     LUMON       Int    See IOPT(14) as described in driver GIANT
C     MPRTIM      Int    See IOPT(19) as described in driver GIANT
C     NRLUSE      Int    Gets the amount of real workspace used
C                        by the iterative linear solver
C     NILUSE      Int    Gets the amount of integer workspace used
C                        by the iterative linear solver
C     NLINOR      Int    Count of linear solver iterations (ordinary
C                        Newton correction computations only)
C     NAMCF       Int    Count of violations of the accuracy matching 
C                        condition for the hk computation.
C     IFAIL       Int    Gets the error return indicator of the
C                        iterative linear solver
C     QLIINI      Logic  See internal variables description in NJINT
C     ------------------------------------------------------------
C*    End Prologue
      INTRINSIC DMAX1,DMIN1,DSQRT
      EXTERNAL NJITSL
      INTEGER J,L1
      DOUBLE PRECISION HALF,ONE,EPSIN,SH,DN,ZERO,
     $                 DIFNRM,EPSH,FCDNM,DMYPRI
      PARAMETER (HALF=0.5D0,ONE=1.0D0,ZERO=0.0D0)
      LULSOL=ITOPT(22)
      DN=DBLE(FLOAT(N))
      MAXL1=LITMAX
      IF (NITER.NE.0) THEN
        DO 10 L1=1,N
          DX(L1)  = -DXQ(L1)
          DELX(L1) = ZERO
10      CONTINUE
      ENDIF
      RWK(NRWLAI)=RWK(41)
      EPSK=RHO/(ONE+RHO)
      IF (MPRMON.GE.4) WRITE(LUMON,30002) HPOST
      ITOPT(1)=0
      IF (NONLIN.NE.1) THEN
        ITOPT(33)=IFIN
        EPSIN=EPSK/(ONE+EPSK)
      ELSE
        ITOPT(33)=1
        EPSIN=RTOL
      ENDIF
      IF (MPRMON.GE.4) THEN
        WRITE(LUMON,10001) EPSK,EPSIN
10001   FORMAT(' +++ EPSK := ',D10.3,2X,'EPS := ',D10.3,' +++',/,
     $         ' Starting a priori ordinary Newton-correction comp.')
      ENDIF
      EPSH=EPSIN
      IF (LULSOL.GT.0) WRITE(LULSOL,10002) NITER
10002 FORMAT(' A priori ordinary correction; niter=',I3)
      IF (MPRTIM.NE.0) CALL MONON (4)
      CALL NJITSL(N,MAXL1,EPSIN,XWI,MULJAC,PRECON,ITSOL,
     $            DX,DELX,F,ITOPT,
     $            LRWLI,RWK(NRWLAI),NRLUSE,
     $            LIWLI,IWK(NIWLAI),NILUSE,
     $            LIWKU,IWKU,LRWKU,RWKU,
     $            DELNRM,SOLNRM,DIFNRM,IFAIL)
      IF (MPRTIM.NE.0) CALL MONOFF(4)
      IF (IFAIL.NE.0) GOTO 9999
      IF (.NOT.QLIINI) THEN
        NRWKFR = NRWKFR+NRLUSE
        NIWKFR = NIWKFR+NILUSE
C       Store lengths of actually required workspaces
        IWK(19) = NRWKFR-1
        IWK(18) = NIWKFR-1
        QLIINI = .TRUE.
      ENDIF
      ITERM=ITOPT(50)
      SH=SOLNRM
      DELNRM = ZERO
      SOLNRM = ZERO
      DO 20 J=1,N
        DELNRM = DELNRM + (DELX(J)/XW(J))**2
        SOLNRM = SOLNRM + (DX(J)/XW(J))**2
20    CONTINUE
      DELNRM = DSQRT(DELNRM)
      SOLNRM = DSQRT(SOLNRM)
      IF (MPRMON.GE.4)
     $  WRITE(LUMON,20001) 'A priori iteration',MAXL1,EPSIN
20001 FORMAT(' +++ Std. Newton correcture: ',A,' +++',/,2X,
     $       '#Iter = ',I4,2X,'Est. Rel. prec. = ',D10.3,/,
     $       ' ++++++++++')
      IF ( NITER.NE.0 .AND. MAXL1.NE.0 ) THEN 
        HPOST=(DELNRM*SOLNRM)/(FCA*DSQRT(FCNUMP))
        IF (MPRMON.GE.4) WRITE(LUMON,30002) HPOST
30002   FORMAT(' Hk(Est) = ',D18.10)
        FCDNM = (DELNRM*SOLNRM)**2
        IF(FCDNM.GT.FCNUMP*FCMIN2 .OR.
     $     (NONLIN.EQ.4 .AND. FCA**2*FCNUMP .LT. 4.0D0*FCDNM)) THEN
          DMYPRI = FCA*DSQRT(FCNUMP/FCDNM)
        ELSE
          DMYPRI = ONE/FCMIN
        ENDIF
        FC = DMIN1(DMYPRI,ONE)
        IF (NONLIN.EQ.4) FC = DMIN1(HALF*DMYPRI,ONE)
      ENDIF
      EPSK=RHO*DMIN1(ONE/(ONE+RHO),HPOST) 
      EPSINK=DMAX1(EPSK/(ONE+EPSK),RTOL*RTOL)
C     No apost. iteration, if lin. solver already found the solution
      IF (EPSIN .LE. EPSINK*RTOL .OR. NONLIN.EQ.1) ITERM=0
      IF ( (FC.EQ.ONE .AND. ITERM.NE.0) .OR. ITERM.EQ.2 ) THEN
        EPSIN=EPSINK
        ITOPT(1)=1
        ITOPT(33)=1
        MAXL1=LITMAX
        IF (MPRMON.GE.4) THEN
          WRITE(LUMON,30005) EPSIN
30005     FORMAT(' +++ EPS := ',D10.3,' +++',/,
     $         ' Starting a post. ordinary Newton-correction comp.')
        ENDIF
        EPSH=EPSIN
        IF (LULSOL.GT.0) WRITE(LULSOL,30006) NITER
30006   FORMAT(' A posteriori ordinary correction; niter=',I3)
        IF (MPRTIM.NE.0) CALL MONON (4)
        CALL NJITSL(N,MAXL1,EPSIN,XWI,MULJAC,PRECON,ITSOL,
     $              DX,DELX,F,ITOPT,
     $              LRWLI,RWK(NRWLAI),NRLUSE,
     $              LIWLI,IWK(NIWLAI),NILUSE,
     $              LIWKU,IWKU,LRWKU,RWKU,
     $              DELNRM,SOLNRM,DIFNRM,IFAIL)
        IF (MPRTIM.NE.0) CALL MONOFF(4)
        IF (IFAIL.NE.0) GOTO 9999
        DELNRM = ZERO
        SOLNRM = ZERO
        DO 40 J=1,N
          DELNRM = DELNRM + (DELX(J)/XW(J))**2
          SOLNRM = SOLNRM + (DX(J)/XW(J))**2
40      CONTINUE
        DELNRM = DSQRT(DELNRM)
        SOLNRM = DSQRT(SOLNRM)
        IF (MPRMON.GE.4) 
     $    WRITE(LUMON,20001) 'A post. iteration',MAXL1,EPSIN
        IF (NITER.NE.0.AND.MPRMON.GE.4) THEN 
          HPOST=(DELNRM*SOLNRM)/(FCA*DSQRT(FCNUMP))
          WRITE(LUMON,30002) HPOST
          EPSK=RHO*DMIN1(ONE/(ONE+RHO),HPOST) 
          EPSINK=DMAX1(EPSK/(ONE+EPSK),RTOL*RTOL) 
          IF (EPSIN .GT. EPSINK) THEN
            NAMCF = NAMCF+1
            WRITE(LUMON,*) ' +++ A second aposteriori computation ',
     $                     ' may be needed !!! +++'
          ENDIF
        ENDIF
      ENDIF
      EPSINK = EPSIN/(ONE-EPSIN)
      EPSK = EPSH
9999  CONTINUE
      NLINOR=NLINOR+MAXL1
C     End of subroutine NJITS1
      RETURN
      END
C
      SUBROUTINE NJSCAL(N,X,XA,XSCAL,XW,ISCAL,QINISC,IOPT,LRWK,RWK)
C*    Begin Prologue SCALE
      INTEGER N
      DOUBLE PRECISION X(N),XSCAL(N),XA(N),XW(N)
      INTEGER ISCAL
      LOGICAL QINISC
      INTEGER IOPT(50),LRWK
      DOUBLE PRECISION RWK(LRWK)
C     ------------------------------------------------------------
C
C*    Summary :
C    
C     S C A L E : To be used in connection with GIANT .
C       Computation of the internal scaling vector XW used for the
C       Jacobian matrix, the iterate vector and it's related
C       vectors - especially for the solution of the linear system
C       and the computations of norms to avoid numerical overflow.
C
C*    Input parameters
C     ================
C
C     N         Int     Number of unknowns
C     X(N)      Dble    Actual iterate
C     XA(N)     Dble    Previous iterate
C     XSCAL(N)  Dble    User scaling passed from parameter XSCAL
C                       of interface routine GIANT
C     ISCAL     Int     Option ISCAL passed from IOPT-field
C                       (for details see description of IOPT-fields)
C     QINISC    Logical = .TRUE.  : Initial scaling
C                       = .FALSE. : Subsequent scaling
C     IOPT(50)  Int     Options array passed from GIANT parameter list
C     LRWK      Int     Length of real workspace
C     RWK(LRWK) Dble    Real workspace (see description above)
C
C*    Output parameters
C     =================
C
C     XW(N)     Dble   Scaling vector computed by this routine
C                      All components must be positive. The follow-
C                      ing relationship between the original vector
C                      X and the scaled vector XSCAL holds:
C                      XSCAL(I) = X(I)/XW(I) for I=1,...N
C
C*    Subroutines called: ZIBCONST
C
C*    Machine constants used
C     ======================
C
      DOUBLE PRECISION EPMACH,SMALL
C
C     ------------------------------------------------------------
C*    End Prologue
      INTRINSIC DABS,DMAX1
      DOUBLE PRECISION HALF
      PARAMETER (HALF=0.5D0)
      INTEGER MPRMON,LUMON
      CALL ZIBCONST(EPMACH,SMALL)
C*    Begin
      DO 1 L1=1,N
        IF (ISCAL.EQ.1) THEN
          XW(L1) = XSCAL(L1)
        ELSE
          XW(L1)=DMAX1(XSCAL(L1),(DABS(X(L1))+DABS(XA(L1)))*HALF,SMALL)
        ENDIF
1     CONTINUE
C$Test-Begin
      MPRMON = IOPT(13)
      IF (MPRMON.GE.6) THEN
        LUMON = IOPT(14)
        WRITE(LUMON,*) ' '
        WRITE(LUMON,*) ' ++++++++++++++++++++++++++++++++++++++++++'
        WRITE(LUMON,*) '      X-components   Scaling-components    '
        WRITE(LUMON,10) (X(L1), XW(L1), L1=1,N)
10      FORMAT('  ',D18.10,'  ',D18.10)
        WRITE(LUMON,*) ' ++++++++++++++++++++++++++++++++++++++++++'
        WRITE(LUMON,*) ' '
      ENDIF
C$Test-End
C     End of subroutine NJSCAL
      RETURN
      END
C
      SUBROUTINE NJITSL(N,MAXIT,EPS,XW,MULJAC,PRECON,ITSOL,
     $                 SOL,DEL,RHS,IOPT,
     $                 LRWK,RWK,NRW,LIWK,IWK,NIW,
     $                 LIWKU,IWKU,LRWKU,RWKU,
     $                 DELNRM,SOLNRM,DIFNRM,IERR)
C*    Begin Prologue NJITSL
      INTEGER N,MAXIT
      DOUBLE PRECISION EPS,XW(N)
      EXTERNAL MULJAC,PRECON,ITSOL
      DOUBLE PRECISION SOL(N),DEL(N),RHS(N)
      INTEGER IOPT(50)
      INTEGER LRWK
      DOUBLE PRECISION RWK(LRWK)
      INTEGER NRW
      INTEGER LIWK
      INTEGER IWK(LIWK)
      INTEGER NIW
      INTEGER LIWKU
      INTEGER IWKU(LIWKU)
      INTEGER LRWKU
      DOUBLE PRECISION RWKU(LRWKU)
      DOUBLE PRECISION DELNRM,SOLNRM,DIFNRM
      INTEGER IERR
C     ------------------------------------------------------------
C
C*    Summary :
C
C     N J I T S L - Iterative solution of a linear system (implicitly
C                   given matrix by subroutine MULJAC -
C                   computes A*vector)
C
C*    Parameters list description
C     ===========================
C    
C*    External subroutines
C     ====================
C
C     MULJAC,PRECON,ITSOL see description in driver subroutine
C                         GIANT  
C
C*    Input parameters (* marks inout parameters)
C     ===========================================
C
C     N          Int     Number of variables and nonlinear equations
C   * MAXIT      Int     Maximum number of iterations allowed   
C     EPS        Dble    Required relative precision
C     XW(N)      Dble    Scaling vector
C   * SOL(N)     Dble    The iteration starting vector
C     RHS(N)     Dble    The right hand side vector
C     TRUE(N)    Dble    The true solution of the linear system
C                        (for testwise comparation purposes)
C     IOPT(50)   Int     Options array (see below for usage)
C
C*    Output parameters
C     =================
C
C     MAXIT      Int     Number of iterations done  
C     EPS        Dble    Required relative precision
C     SOL(N)     Dble    The solution or final iterate vector
C     DEL(N)     Dble    The difference SOLfinal(N)-SOLstart(N)
C     DELNRM     Dble    The norm2 of SOLfinal(N)-SOLstart(N) 
C     SOLNRM     Dble    The norm2 of SOLfinal(N)
C     DIFNRM     Dble    The norm2 of the final correction of SOL(N)
C     IERR       Int     Error code - a zero return signals ok.
C
C*    Workspace parameters
C     ====================
C
C     LRWK           Int     Declared dimension of real workspace
C                            Required minimum: see specific linear
C                                              solver
C     RWK(LRWK)      Dble    Real Workspace
C     NRW            Int     (Out) Real Workspace used
C     LIWK           Int     Declared dimension of integer workspace
C                            Required minimum: see specific linear
C                                              solver
C     IWK(LIWK)      Int    Integer Workspace
C     NIW            Int    (Out) Integer Workspace used
C     LIWKU          Int    Declared dimension of user function 
C                           integer workspace.
C     IWKU(LIWKU)    Int    User function integer workspace
C     LRWKU          Int    Declared dimension of user function 
C                           real workspace.
C     RWKU(LRWKU)    Dble   User function real workspace
C
C     Usage of options array IOPT :
C     =============================
C
C     Pos. Name            Meaning
C
C       1  QSUCC           Indicator for the iteration modus:
C                          =0 : A new iteration will be started
C                          =1 : A previously terminated iteration will 
C                               be continued - recommended to be done
C                               without some special restart.
C                               (usally with a smaller tolerance RTOL
C                                prescribed as before).
C       2..4               Reserved
C       5                  The iterative linear solver to be used
C                          =1: Good Broyden (GBIT1I)
C                          =9: Other linear solver ITSOL supplied by
C                              user via the GIANT parameter ITSOL 
C       6..12              Reserved
C      13  MPRLIN          Output level for the iterative linear solver 
C                          monitor
C                          Is interpreted by Good Broyden as follows:
C                          = 0 : no output will be written
C                          = 1 : only a summary output will be written
C                          > 1 : reserved for future use
C                          =-j : A summary output and additionally each 
C                                j-th iterates statistics will be 
C                                written
C      14  LULIN           Logical unit number for print monitor
C      15..16              Reserved
C      17  MPROPT          Print monitor option:
C                          Is interpreted by Good Broyden as follows:
C                          = 0: Standard print monitor
C                          = 1: Test print monitor for special purposes
C      18                  Reserved
C      19  MPRTIM          Output level MPRTIM for the time monitor.
C                          See IOPT(19) of driver subroutine GIANT 
C      20..21              Reserved
C      22  LUSPE           Logical output unit for special information
C                          (zero means no output by Good Broyden)
C      23..30              Reserved
C      31  KMAX            Maximum number of latest iterates to be saved
C                          and used by the iterative linear solver Good
C                          Broyden. May be interpreted by a user
C                          supplied linear solver in a simular way.
C                          See IOPT(41) in driver subroutine GIANT for 
C                          details.
C      32                  Reserved
C      33  IFINI           Type of stop criterium to be used by GBIT1I:
C                          =1: stop, if relcorr(X) < TOL
C                          =2: stop, if relcorr(X) < TOL or 
C                                    if relcorr(XDEL) < 2*TOL
C      34..40              Reserved
C      41  LITER           Output from GBIT1I: Number of iteration steps
C                          totally done
C      42  K               Output from GBIT1I: Number of iteration steps 
C                          done since latest internal restart due to
C                          restrictions implied by KMAX, TAUMIN and
C                          TAUMAX.
C      43  NMULJ           Output from GBIT1I: Number of (pairwise)
C                          calls of MULJAC and PRECON 
C      44..49              Reserved  
C      50  IFIOUT          On output from GBIT1I, this field contains 
C                          the information, which criterium leads to
C                          convergence stop  (if on input 2):
C                          =1: if relcorr(X) < EPS has been satisfied
C                          =2: relcorr(DEL) < 2*EPS has been satisfied
C
C     ------------------------------------------------------------
C*    End Prologue
      DOUBLE PRECISION ERR
      INTEGER L
      PARAMETER(L=3)
      LOGICAL QCONT
      EXTERNAL GBIT1I
      IERR  = 0
      IFINI = IOPT(33)
      LTYP  = IOPT(5)
      KMAX = IOPT(31)
      IF (KMAX.LE.-2) KMAX=0
      QCONT = IOPT(1).EQ.1
      IPRINT=IOPT(13)
      IPUNIT=IOPT(14)
C
      IF (LTYP.EQ.1) THEN
        L1 = 11
        L2 = L1 + N*KMAX
        L4 = L2 + N
        L5 = L4 + N
        L6 = L5 + N
        L61 = L6 + N
        L7 = L61 + KMAX
        L8 = L7 + KMAX
        L9 = L8 + L
        L10 = L9 + 4
        NRW = L10 - 1
        NIW = 0
        IF (IPRINT.LT.0) THEN
          WRITE(IPUNIT,1000) KMAX 
1000      FORMAT(6X,'KMax = ',I5)     
        ENDIF
        IF (NRW.GT.LRWK) THEN
          IERR = -10
          IF (IPRINT.NE.0) WRITE(IPUNIT,10000) 'Real',NRW-LRWK
10000     FORMAT(' NJITSL - ',A,' Workspace exhausted,',
     $           ' at least more required: ',I6)
          RETURN
        ENDIF
C       Special Newton print format option
        IOPT(17)=1
        IDMUL = 5
        IDPRE = 3
C
        CALL GBIT1I(N,MAXIT,EPS,XW,MULJAC,PRECON,
     $              SOL,DEL,RHS,IOPT,QCONT,IERR,
     $              KMAX,RWK(L1),RWK(L2),RWK(L4),RWK(L5),
     $              RWK(L6),RWK(L61),RWK(L7),IOPT(41),IOPT(42),
     $              L,RWK(L8),RWK(L9),RWK(L9+1),RWK(L9+2),RWK(L9+3),
     $              LIWKU,IWKU,LRWKU,RWKU,DELNRM,SOLNRM,DIFNRM,
     $              RWK(1),RWK(2),RWK(3),RWK(4),IOPT(43),IDMUL,
     $              IDPRE)
        MAXIT = IOPT(41)
      ENDIF
      IF (LTYP.EQ.9) THEN
        CALL ITSOL( N, RHS, SOL, DEL, XW, MULJAC, PRECON, 
     $                EPS, MAXIT, ITER, ERR, IERR,
     $                IOPT, LRWK, RWK, NRW, LIWK, IWK, NIW, 
     $                LRWKU, RWKU, LIWKU, IWKU )
        EPS   = ERR
        MAXIT = ITER
        IF (IERR.GT.0) RETURN
        SOLNRM=0.0D0
        DELNRM=0.0D0
        DO 10 J=1,N
          SOLNRM = SOLNRM + (SOL(J)/XW(J))**2
          DELNRM = DELNRM + (DEL(J)/XW(J))**2
10      CONTINUE
      ENDIF
C 
      IOPT(5) = LTYP
C
C     End of subroutine NJITSL
      RETURN
      END
C
      SUBROUTINE NJPRV1(DLEVF,DLEVX,FC,NITER,MPRMON,LUMON,QMIXIO)
C*    Begin Prologue NJPRV1
      DOUBLE PRECISION DLEVF,DLEVX,FC
      INTEGER NITER,MPRMON,LUMON
      LOGICAL QMIXIO
C     ------------------------------------------------------------
C
C*    Summary :
C
C     N 1 P R V 1 : Printing of intermediate values (Type 1 routine)
C
C*    Parameters
C     ==========
C
C     DLEVF, DLEVX   See descr. of internal double variables of NJINT
C     FC,NITER,MPRMON,LUMON
C                  See parameter descr. of subroutine NJINT
C     QMIXIO Logical  = .TRUE.  , if LUMON.EQ.LUSOL
C                     = .FALSE. , if LUMON.NE.LUSOL
C
C     ------------------------------------------------------------
C*    End Prologue
C     Print Standard - and natural level
      IF(QMIXIO)THEN
1       FORMAT(2X,66('*'))
        WRITE(LUMON,1)
2       FORMAT(8X,'It',7X,'Normf ',10X,'Normx ')
        IF (MPRMON.GE.3) WRITE(LUMON,2)
3       FORMAT(8X,'It',7X,'Normf ',10X,'Normx ',8X,'Damp.Fct.')
        IF (MPRMON.EQ.2) WRITE(LUMON,3)
      ENDIF
4     FORMAT(6X,I4,5X,D10.3,2X,4X,D10.3)
      IF (MPRMON.GE.3.OR.NITER.EQ.0) 
     $  WRITE(LUMON,4) NITER,DLEVF,DLEVX
5     FORMAT(6X,I4,5X,D10.3,6X,D10.3,6X,F7.5)
      IF (MPRMON.EQ.2.AND.NITER.NE.0) 
     $  WRITE(LUMON,5) NITER,DLEVF,DLEVX,FC
      IF(QMIXIO)THEN
6       FORMAT(2X,66('*'))
        WRITE(LUMON,6)
      ENDIF
C     End of subroutine NJPRV1
      RETURN
      END
C
      SUBROUTINE NJPRV2(DLEVF,DLEVX,FC,NITER,MPRMON,LUMON,QMIXIO,
     $                  CMARK)
C*    Begin Prologue NJPRV2
      DOUBLE PRECISION DLEVF,DLEVX,FC
      INTEGER NITER,MPRMON,LUMON
      LOGICAL QMIXIO
      CHARACTER*1 CMARK
C     ------------------------------------------------------------
C
C*    Summary :
C
C     N 1 P R V 2 : Printing of intermediate values (Type 2 routine)
C
C*    Parameters
C     ==========
C
C     DLEVF, DLEVX   See descr. of internal double variables of N2INT
C     FC,NITER,MPRMON,LUMON
C                  See parameter descr. of subroutine N2INT
C     QMIXIO Logical  = .TRUE.  , if LUMON.EQ.LUSOL
C                     = .FALSE. , if LUMON.NE.LUSOL
C     CMARK Char*1    Marker character to be printed before DLEVX
C
C     ------------------------------------------------------------
C*    End Prologue
C     Print Standard - and natural level, and damping
C     factor
      IF(QMIXIO)THEN
1       FORMAT(2X,66('*'))
        WRITE(LUMON,1)
2       FORMAT(8X,'It',7X,'Normf ',10X,'Normx ',8X,'Damp.Fct.')
        WRITE(LUMON,2)
      ENDIF
3     FORMAT(6X,I4,5X,D10.3,4X,A1,1X,D10.3,2X,4X,F7.5)
      WRITE(LUMON,3)NITER,DLEVF,CMARK,DLEVX,FC
      IF(QMIXIO)THEN
4       FORMAT(2X,66('*'))
        WRITE(LUMON,4)
      ENDIF
C     End of subroutine NJPRV2
      RETURN
      END
C
      SUBROUTINE NJSOUT(N,X,MODE,IOPT,RWK,NRW,IWK,NIW,RWKU,IWKU,
     $                  MPRINT,LUOUT)
C*    Begin Prologue SOLOUT
      INTEGER N
      DOUBLE PRECISION X(N)
      INTEGER NRW
      INTEGER MODE
      INTEGER IOPT(50)
      DOUBLE PRECISION RWK(NRW)
      INTEGER NIW
      INTEGER IWK(NIW)
      DOUBLE PRECISION RWKU(*)
      INTEGER IWKU(*)
      INTEGER MPRINT,LUOUT
C     ------------------------------------------------------------
C
C*    Summary :
C
C     S O L O U T : Printing of iterate (user customizable routine)
C
C*    Input parameters
C     ================
C
C     N         Int Number of equations/unknowns
C     X(N)   Dble   iterate vector
C     MODE          =1 This routine is called before the first
C                      Newton iteration step
C                   =2 This routine is called with an intermedi-
C                      ate iterate X(N)
C                   =3 This is the last call with the solution
C                      vector X(N)
C                   =4 This is the last call with the final, but
C                      not solution vector X(N)
C     IOPT(50)  Int The option array as passed to the driver
C                   routine(elements 46 to 50 may be used
C                   for user options)
C     MPRINT    Int Solution print level 
C                   (see description of IOPT-field MPRINT)
C     LUOUT     Int the solution print unit 
C                   (see description of see IOPT-field LUSOL)
C
C
C*    Workspace parameters
C     ====================
C
C     NRW, RWK, NIW, IWK    see description in driver routine
C
C*    Use of IOPT by this routine
C     ===========================
C
C     Field 46:       =0 Standard output
C                     =1 GRAZIL suitable output
C
C     ------------------------------------------------------------
C*    End Prologue
      LOGICAL QGRAZ,QNORM
C*    Begin
      QNORM = IOPT(46).EQ.0
      QGRAZ = IOPT(46).EQ.1
      IF(QNORM) THEN
1        FORMAT('  ',A,' data:',/)
         IF (MODE.EQ.1) THEN
101        FORMAT('  Start data:',/,'  N =',I5,//,
     $            '  Format: iteration-number, (x(i),i=1,...N), ',
     $            'Normf , Normx ',/)
           WRITE(LUOUT,101) N
           WRITE(LUOUT,1) 'Initial'
         ELSE IF (MODE.EQ.3) THEN
           WRITE(LUOUT,1) 'Solution'
         ELSE IF (MODE.EQ.4) THEN
           WRITE(LUOUT,1) 'Final'
         ENDIF
2        FORMAT(' ',I5)
C        WRITE          NITER
         WRITE(LUOUT,2) IWK(1)
3        FORMAT((12X,3(D18.10,1X)))
         WRITE(LUOUT,3)(X(L1),L1=1,N)
C        WRITE          DLEVF,   DLEVX
         WRITE(LUOUT,3) RWK(19),DSQRT(RWK(18)/DBLE(FLOAT(N)))
         IF(MODE.EQ.1.AND.MPRINT.GE.2) THEN
           WRITE(LUOUT,1) 'Intermediate'
         ELSE IF(MODE.GE.3) THEN
           WRITE(LUOUT,1) 'End'
         ENDIF
      ENDIF
      IF(QGRAZ) THEN
        IF(MODE.EQ.1) THEN
10        FORMAT('&name com',I3.3,:,255(7(', com',I3.3,:),/))
          WRITE(LUOUT,10)(I,I=1,N+2)
15        FORMAT('&def  com',I3.3,:,255(7(', com',I3.3,:),/))
          WRITE(LUOUT,15)(I,I=1,N+2)
16        FORMAT(6X,': X=1, Y=',I3)
          WRITE(LUOUT,16) N+2
        ENDIF
20      FORMAT('&data ',I5)
C        WRITE          NITER
        WRITE(LUOUT,20) IWK(1) 
21      FORMAT((6X,4(D18.10)))
        WRITE(LUOUT,21)(X(L1),L1=1,N)
C        WRITE          DLEVF,   DLEVX
        WRITE(LUOUT,21) RWK(19),DSQRT(RWK(18)/DBLE(FLOAT(N)))
        IF(MODE.GE.3) THEN
30        FORMAT('&wktype 3111',/,'&atext x ''iter''')
          WRITE(LUOUT,30)
35        FORMAT('&vars = com',I3.3,/,'&atext y ''x',I3,'''',
     $           /,'&run')
          WRITE(LUOUT,35) (I,I,I=1,N)
36        FORMAT('&vars = com',I3.3,/,'&atext y ''',A,'''',
     $           /,'&run')
          WRITE(LUOUT,36) N+1,'Normf ',N+2,'Normx '
C39       FORMAT('&stop')
C         WRITE(LUOUT,39)
        ENDIF
      ENDIF
C     End of subroutine NJSOUT
      RETURN
      END
C*    End package
C
      SUBROUTINE NJGOUT(LUGOUT,NITER,NRED,DLEVXA,DLEVX,FC,ITORD,ITSIM,
     $                  EPSORD,ITERM)
      INTEGER LUGOUT,NITER,NRED
      DOUBLE PRECISION DLEVXA,DLEVX,FC,ITORD,ITSIM,EPSORD,ITERM
      IF (LUGOUT.GT.0) 
     $  WRITE(LUGOUT,10000) NITER,NRED,DLEVXA,DLEVX,FC,ITORD,ITSIM,
     $                      EPSORD,ITERM
10000 FORMAT(I4,1X,I1,7(1X,D10.3))
      RETURN
      END
