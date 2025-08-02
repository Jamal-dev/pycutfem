C*    Begin Prologue
C     ------------------------------------------------------------
C
C*  Title
C
C     SLAPInt - An easy-to-use supplement package for the GIANT program.
C
C*  Written by        U. Nowak, L. Weimann
C*  Purpose           An easy-to-use package for the GIANT program.
C*  Category          F2a. - Systems of nonlinear equations
C*  Version           1.0.1
C*  Revision          April 1996
C*  Latest Change     April 1996
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
C  
C     SLAPInt provides a facility to use the GIANT program without
C     the need of programming the Jacobian times vector subroutine
C     MULJAC and preconditioner subroutine PRECON - instead these
C     subroutines will be replaced by subroutines of this package
C     which may call standard subroutines from the SLAP (Sparse
C     Linear Algebra) package or from LINPACK. The SLAPInt package
C     performs these operations on a matrix (i.e. the Jacobian)
C     provided in the SLAP Triad or SLAP Column format.     
C     Using this package, and the GIANT program with it's standard
C     linear solver "Good Broyden", the only users task is to provide
C     his problem function subroutine and the appropriate Jacobian
C     subroutine, which generates the Jacobian in the SLAP Triad or
C     SLAP Column format.
C     ------------------------------------------------------------
C
C*    Usage:
C     
C     To use this package, the user must regard the following name 
C     conventions:
C     His problem function subroutine must be named FCN1;
C     his Jacobian subroutine must be named JAC1;
C     In the main program which calls the GIANT subroutine, must be
C     declared the external names EFCN, EJAC, MULJAC and PRECON, and
C     these names must be passed in the actual parameter calling list
C     of the GIANT subroutine at the equal named formal parameter
C     positions.
C     Additionally, IOPT(4) must be set to 0 (or 1) to use the
C     standard linear solver of the GIANT program.
C     Furthermore, the SLAPInt package needs a real workspace named 
C     rwku and an integer workspace named iwku to store information.
C     Parts of these workspaces may be also used by the user to store
C     data needed for calculations in his subroutines FCN1 and JAC1.
C     These workspaces must be passed in the actual GIANT parameter
C     call list at the equal named formal parameter positions.
C
C     The workspaces are subdivided by the SLAPInt package as follows:
C
C     The first ten positions of the integer workspace iwku are 
C     reserved for control information passed to the SLAPInt package, 
C     such as lengths of workspaces needed for user data or selection
C     of a preconditioner. For details, refer to description of first
C     ten iwku positions below. Following, the user may store his data
C     needed for FCN1 and JAC1 starting at position iwku(11) up to the
C     position iwku(10+nusiwk), where nusiwk denotes the amount of
C     integer user workspace needed for user purpose. Following,
C     the Jacobian row indices information is stored, starting at
C     position iwku(11+nusiwk) and up to iwku(10+nusiwk+nzmax)
C     (where nzmax denotes the maximum number of nonzero elements
C      which the Jacobian may have). Next, the column indices or 
C     column pointers of the Jacobian are stored, starting at 
C     iwku(11+nusiwk+nzmax) and up to iwku(10+nusiwk+2*nzmax) respec-
C     tive iwku(10+nusiwk+nzmax+n+1). See description of possible
C     Jacobian formats below. The SLAP subroutines always work with
C     the SLAP column (pointers) format, so if the user provides the
C     Jacobian in the first named SLAP format, it's converted to the
C     second one. Therefore, any integer workspace needed for the
C     preconditioner subroutines starts at iwku(11+nusiwk+nzmax+n+1).
C
C     The real workspace rwku may be used for reference by FCN1 and
C     JAC1 starting at position rwku(1) up to rwku(nusrwk), where
C     the length nusrwk is supplied by the user. Next, starting at
C     position rwku(nusrwk+1) up to rwku(nusrwk+nzmax), the real values
C     of the Jacobian are stored by the SLAPInt package. The remaining
C     upper workspace is used by the package as workspace for the
C     preconditioner subroutines.
C
C     The total amount of workspace needed by the SLAPInt package
C     depends on the selected preconditioner. It may be calcuted
C     using the following formulas (n denotes the number of nonlinear
C     equations) :
C     rwku : nzmax + lrwpre
C     iwku : max( 10+2*nzmax , 10+nzmax+n+1+liwpre )
C     where lrwpre and liwpre must be choosen dependent on the selected
C     preconditioner as follows :
C     Incomplete LU-decomposition: lrwpre = nzmax
C                                  liwpre = nzmax+3*n+2
C     Lower triangle:              lrwpre = int( (nzmax+n)/2)
C                                  liwpre = int( (nzmax+n)/2) + n+1
C     Diagonal scaling:            lrwpre = n , liwpre = 0 
C     Block diagonal scaling:      lrwpre = n*diagonal_block_size
C                                  liwpre = lrwpre
C     No preconditioner:           lrwpre = liwpre = 0
C
C     Remember, to get the total sizes of rwku and iwku, you must
C     add to the sizes computed above the amounts for your own data
C     stored or referenced by FCN1 and JAC1.
C
C
C*    Parameter description of user subroutines FCN1 and JAC1:
C
C
C     Subroutine heading of FCN1:
C       subroutine fcn1(n,x,f,rwku,iwku,nfcn,ifail) 
C       integer n, nfcn, ifail
C       double precision x(n), f(n)
C       double precision rwku(*)
C       integer iwku(*)
C       ...
C
C     Parameters of FCN1:
C         n         int    number of vector components (input)
C         x(n)      double vector of unknowns (input)
C         f(n)      double vector of function values (output)
C         rwku(*)   double real workspace for the user function -
C                          passed to FCN1 from the formal parameter 
C                          rwku of the driver subroutine GIANT at 
C                          starting position rwku(1).
C         iwku(*)   int    integer workspace for the user function -
C                          passed to FCN1 from the formal parameter 
C                          iwku of the driver subroutine GIANT at
C                          starting position iwku(11).
C         nfcn      int    Count of FCN1 calls (input). must not
C                          be altered by FCN1.
C         ifail     int    FCN1 evaluation-failure indicator (output).
C                          Has always value 0 (zero) on input.
C                          indicates failure of FCN1 evaluation
C                          and causes termination of GIANT,
C                          if set to a negative value on output.
C     
C
C     Subroutine heading of JAC1:
C       subroutine jac1(fcn,n,x,xw,f,nzmax,idummy,a,ia,ja,nfill,
C      $                rwku,iwku,njac,ifail)
C       external fcn
C       integer n
C       double precision x(n),xw(n),f(n)
C       integer nzmax,idummy
C       double precision a(nzmax)
C       integer ia(nzmax),ja(nzmax)
C       integer nfill
C       double precision rwku(*)
C       integer iwku(*)
C       integer njac,ifail
C       ... 
C
C     Parameters of JAC1:
C         fcn        ext    the problem function (subroutine) 
C                           reference
C         n          int    number of vector components (input)
C         x(n)       double vector of unknowns (input)
C         f(n)       double vector of function values as supplied
C                           by subroutine fcn with same input
C                           argument x(n) (input)
C         nzmax      int    length of arrays a, ia, ja
C         idummy            (dummy argument - for backward
C                            compatibility)
C         a(nzmax)   double array to get the real values of the
C                           Jacobian in a SLAP format. 
C                           May be changed by the SLAPInt package -
C                           see note below! 
C         ia(nzmax)  int    array to get the row indices of the
C                           Jacobian in a SLAP format.
C                           May be changed by the SLAPInt package -
C                           see note below! 
C         ja(nzmax)  int    array to get the column indices (SLAP
C                           triad format) or column pointers (SLAP
C                           column format) of the Jacobian.
C                           May be changed by the SLAPInt package -
C                           see note below! 
C         nfill      int    number of elements used of arrays
C                           a, ia, ja (output).
C                           0 < nfill <= nzmax is required.
C         rwku(*)   double real workspace for the user function -
C                          passed to FCN1 from the formal parameter 
C                          rwku of the driver subroutine GIANT at 
C                          starting position rwku(1).
C         iwku(*)   int    integer workspace for the user function -
C                          passed to FCN1 from the formal parameter 
C                          iwku of the driver subroutine GIANT at
C                          starting position iwku(11).
C         njac       int    count of JAC1 calls (input). must not
C                           be altered by JAC1.
C         ifail      int    JAC1 evaluation-failure indicator (output).
C                           Has always value 0 (zero) on input.
C                           Indicates failure of JAC1 evaluation
C                           and causes termination of GIANT,
C                           if set to a negative value on output.
C     
C*    Note:
C     If the Jacobian is supplied by the user in the SLAP Triad
C     format, it will be changed by calling a SLAP utility
C     subroutine from the SLAPInt package to SLAP column format -
C     this implies, that the contents of the arrays a, ia and ja
C     will be changed! If the Jacobian supplied in SLAP column
C     format, the arrays a, ia and ja will remain unmodified.
C
C
C*    Usage of the first 10 positions of iwku as control input
C     to the SLAPInt package:
C
C      1  lrwku   declared length of rwku (real workspace)
C      2  liwku   declared length of iwku (integer workspace)
C      3  nusrwk  length of real (user) workspace needed for user
C                 purposes (real elements of jacobian, followed 
C                 by preconditioner information starts at position
C                 nusrwk+1)
C      4  nusiwk  length of integer (user) workspace needed for user
C                 purposes (starts at iwk(11), and is followed by the
C                 indices (SLAP) information of the Jacobian and pre-
C                 conditioner information, starting at position 
C                 nusiwk+11
C      5  nzmax   maximum number of Jacobian nonzeros
C      6  nfill   actual number of Jacobian nonzeros
C      7  luerr   i/o-unit number for error messages
C      8  mprerr  error message output level
C      9  ipre    type of preconditioner, valid values are:
C                 0: The SLAP preconditioner ILU (Incomplete LU-
C                    decomposition) will be used,
C                 1: The SLAP preconditioner lower triangle 
C                    will be used,
C                 2: The SLAP preconditioner diagonal scaling
C                    will be used,
C                 3: No preconditioner will be used,
C                 4: The ZIB supplied preconditioner block diagonal
C                    scaling will be used.
C     10  nbsize  size of diagonal blocks, if ipre=4 selected
C                 (preconditioner block diagonal scaling)
C
C
C*    Note:
C     All the above described iwku-positions (except position 6 and
C     in case of ipre.ne.4 also except position 10) must be set 
C     by the calling program of GIANT before calling GIANT. These
C     workspace positions are not available to the user supplied
C     subroutines FCN1 and JAC1, but the space of iwku made available
C     to the user subroutines starts at iwku(11).
C
C
C*      Description of the Jacobian SLAP format
C       (as extracted from the SLAP package description):
C       
C       =================== S L A P Triad format ===================
C       This routine requires that the  matrix A be   stored in  the
C       SLAP  Triad format.  In  this format only the non-zeros  are
C       stored.  They may appear in  *ANY* order.  The user supplies
C       three arrays of  length NELT, where  NELT is  the number  of
C       non-zeros in the matrix: (IA(NELT), JA(NELT), A(NELT)).  For
C       each non-zero the user puts the row and column index of that
C       matrix element  in the IA and  JA arrays.  The  value of the
C       non-zero   matrix  element is  placed  in  the corresponding
C       location of the A array.   This is  an  extremely  easy data
C       structure to generate.  On  the  other hand it   is  not too
C       efficient on vector computers for  the iterative solution of
C       linear systems.  Hence,   SLAP changes   this  input    data
C       structure to the SLAP Column format  for  the iteration (but
C       does not change it back).
C       
C       Here is an example of the  SLAP Triad   storage format for a
C       5x5 Matrix.  Recall that the entries may appear in any order.
C
C           5x5 Matrix       SLAP Triad format for 5x5 matrix on left.
C                              1  2  3  4  5  6  7  8  9 10 11
C       |11 12  0  0 15|   A: 51 12 11 33 15 53 55 22 35 44 21
C       |21 22  0  0  0|  IA:  5  1  1  3  1  5  5  2  3  4  2
C       | 0  0 33  0 35|  JA:  1  2  1  3  5  3  5  2  5  4  1
C       | 0  0  0 44  0|
C       |51  0 53  0 55|
C       
C       =================== S L A P Column format ==================
C       This routine requires  that the  matrix  A be  stored in the
C       SLAP Column format.  In this format the non-zeros are stored
C       counting down columns (except for  the diagonal entry, which
C       must appear first in each  "column")  and are stored  in the
C       double precision array A.   In other words,  for each column
C       in the matrix put the diagonal entry in  A.  Then put in the
C       other non-zero  elements going down  the column (except  the
C       diagonal) in order.   The  IA array holds the  row index for
C       each non-zero.  The JA array holds the offsets  into the IA,
C       A arrays  for  the  beginning  of each   column.   That  is,
C       IA(JA(ICOL)),  A(JA(ICOL)) points   to the beginning  of the
C       ICOL-th   column    in    IA and   A.      IA(JA(ICOL+1)-1),
C       A(JA(ICOL+1)-1) points to  the  end of the   ICOL-th column.
C       Note that we always have  JA(N+1) = NELT+1,  where N is  the
C       number of columns in  the matrix and NELT  is the number  of
C       non-zeros in the matrix.
C       
C       Here is an example of the  SLAP Column  storage format for a
C       5x5 Matrix (in the A and IA arrays '|'  denotes the end of a 
C       column):
C       
C       5x5 Matrix      SLAP Column format for 5x5 matrix on left.
C       1  2  3    4  5    6  7    8    9 10 11
C       |11 12  0  0 15|   A: 11 21 51 | 22 12 | 33 53 | 44 | 55 15 35
C       |21 22  0  0  0|  IA:  1  2  5 |  2  1 |  3  5 |  4 |  5  1  3
C       | 0  0 33  0 35|  JA:  1  4  6    8  9   12
C       | 0  0  0 44  0|
C       |51  0 53  0 55|
C  
C     ------------------------------------------------------------
C*    End Prologue
C
      SUBROUTINE EFCN(N,U,URHS,RWU,IWU,NFCN,IFAIL)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INTEGER N, IFAIL
      DOUBLE PRECISION U(N), URHS(N)
      DOUBLE PRECISION RWU(*)
      INTEGER IWU(*)
C
      PARAMETER (LOWI=11)
C
      CALL FCN1(N,U,URHS,RWU,IWU(LOWI),NFCN,IFAIL)
      RETURN
      END
C
      SUBROUTINE EJAC(FCN,N,U,UWGT,F,RWKU,IWKU,NJAC,IFAIL)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      EXTERNAL FCN,FCN1
      INTEGER N
      DOUBLE PRECISION U(N),UWGT(N),F(N)
      DOUBLE PRECISION RWKU(*)
      INTEGER IWKU(*)
      INTEGER NJAC,IFAIL 
C
      PARAMETER (LOWI=11)
C
      LRWKU  = IWKU(1)
      LIWKU  = IWKU(2)
      NUSRWK = IWKU(3)
      NUSIWK = IWKU(4)
      IFAIL = 0
      IF (NJAC.EQ.0) THEN
C       check for sufficient real and integer workspace (rwku and iwku)
        NZMAX = IWKU(5)
        NRW = NUSRWK + NZMAX
        IF (NRW.GT.LRWKU) THEN
          IFAIL = -10
        ENDIF
        NIW = LOWI-1+NUSIWK+2*NZMAX
        IF (NIW.GT.LIWKU) THEN
          IFAIL = IFAIL -20
        ENDIF
        IF (IFAIL.NE.0) GOTO 9999
      ELSE
        NZMAX  = IWKU(5)
      ENDIF
      LOWIND = LOWI+NUSIWK
      NFILL=0
      CALL JAC1( FCN1,N,U,UWGT,F,NZMAX,IDUMMY,
     $                RWKU(NUSRWK+1), IWKU(LOWIND), IWKU(LOWIND+NZMAX),
     $                NFILL,RWKU,IWKU(LOWI),NJAC,IFAIL)
      IF (IFAIL.LT.0) RETURN
      IWKU(6) = NFILL
      IF (NFILL.LE.0 .OR. NFILL.GT.NZMAX) THEN
        IFAIL= -40
        MPRERR=IWKU(8)
        LUERR =IWKU(7)
        IF (LUERR.NE.0 .AND. MPRERR.GT.0)  WRITE(LUERR,10001) NFILL
10001   FORMAT(' bad or too large number of Jacobian nonzeros: ',I7)
        RETURN
      ENDIF
C     convert to SLAP column format
      CALL DS2Y(N, NFILL, IWKU(LOWIND), IWKU(LOWIND+NZMAX),
     $          RWKU(NUSRWK+1), 0)
C     setup of preconditioner
      LOWRP = NUSRWK+NZMAX+1
      LOWIP = LOWIND+NZMAX+N+1
      LRWKP = LRWKU-LOWRP+1
      LIWKP = LIWKU-LOWIP+1
      CALL SLPREI(N, NFILL, RWKU(NUSRWK+1), IWKU(LOWIND),
     $            IWKU(LOWIND+NZMAX), LIWKP, IWKU(LOWIP), LRWKP,
     $            RWKU(LOWRP), IWKU(1), IFAIL)
      IF (IFAIL.GT.0) IFAIL = -IFAIL
9999  RETURN
      END
C
      SUBROUTINE SLPREI(N, NELT, A, IA, JA, 
     $                    LIWK, IWK, LRWK, RWK, IOPT, IFAIL)
C*    begin prologue slprei
      INTEGER N,NELT
      DOUBLE PRECISION A(NELT)
      INTEGER IA(NELT),JA(NELT)
      INTEGER LIWK
      INTEGER IWK(LIWK)
      INTEGER LRWK
      DOUBLE PRECISION RWK(LRWK)
      INTEGER IOPT(20),IFAIL
C     ------------------------------------------------------------
C
C*    summary :
C
C     S L P R E I - call a builtin preconditioner setup subroutine.
C                   workspace management routine - the actual call
C                   of the preconditioner setup is done by subroutine
C                   slpre1
C
C*    input parameters
C     ================
C
C     n          int     number of variables and nonlinear equations
C     nelt       int     number of nonzeros of the jacobian
C                        stored in slap format in the arrays a, ia, ja
C     a(nelt)    double  jacobian nonzero real values
C     ia(nelt)   int     the corresponding row indices to the real
C                        values stored in a
C     ja(nelt)   int     the column pointers (slap column format)
C     lrwk       int     declared dimension of real workspace
C                        required minimum: see specific precond.
C                                          setup subroutine
C     liwk       int     declared dimension of integer workspace
C                        required minimum: see specific precond.
C                                          setup subroutine
C     iwk(liwk)  int     integer workspace. iwk(1) (=ipre) indicates the
C                        preconditioner to be selected:
C                        0 = the slap incomplete lu (ilu) 
C                        1 = the slap lower triangle
C                        2 = the slap diagonal scaling 
C                        other = no preconditioner selected
C     iopt(20)   int     the options (first 20 elements of iwk)
C
C*    output parameters
C     =================
C
C     rwk(lrwk)  double  real workspace holding preconditioning
C                        information
C     iwk(liwk)  int     integer workspace holding preconditioning
C                        information up from iwk(2).
C     ifail      int     if zero, everything is okay.
C                        if positive, it holds the amount of missing
C                        real or integer workspace (if both workspaces
C                        are too small, the amount of missing integer)
C                        if negative, the selected preconditioning
C                        option is invalid for use with the given
C                        matrix of the linear system
C
C     ------------------------------------------------------------
C*    end prologue
      EXTERNAL SLPRE1
      INTEGER NEL, NU
      IFAIL=0
      IPRE=IOPT(9)
      IWK(1)=IPRE
      IF (IPRE.EQ.3) RETURN
      IF (IPRE.LT.0 .OR. IPRE.GT.4) THEN
        IFAIL = -901
        RETURN
      ENDIF
      IF (IPRE.EQ.0 .OR. IPRE.EQ.1 )
     $  CALL SLPREA( N, NELT, A, IA, JA, NEL, NU )
      IF(IPRE.EQ.1) NEL=NEL+N
      LUERR=IOPT(7)
      MPRERR=IOPT(8)
      IF (IPRE.EQ.4) THEN
        M=IOPT(10)
        NEL=N*M
        IWK(2)=M
      ENDIF
      LR1=1
      IF ( IPRE.EQ.0 .OR. IPRE.EQ.1 .OR. IPRE.EQ.4) THEN
        LR2=LR1+NEL
      ELSE
        LR2=LR1
      ENDIF
      IF (IPRE.EQ.0) THEN
        LR3=LR2+NU
      ELSE
        LR3=LR2
      ENDIF
      IF (IPRE.EQ.0 .OR. IPRE.EQ.2) THEN
        LR4=LR3+N
      ELSE
        LR4=LR3
      ENDIF
      NRW=LR4-1
      LI1=2
      IF (IPRE.EQ.0) LI1=4
      IF (IPRE.EQ.1 .OR. IPRE.EQ.4) LI1=3
      IF (IPRE.EQ.0 .OR. IPRE.EQ.1) THEN
        LI2=LI1+NEL
        LI3=LI2+N+1
      ELSE IF (IPRE.EQ.4) THEN
        LI2=LI1+NEL
        LI3=LI2
      ELSE
        LI3=LI1
      ENDIF
      IF (IPRE.EQ.0) THEN
        LI4=LI3+NU
        LI5=LI4+N+1
      ELSE
        LI4=LI3
        LI5=LI4
      ENDIF
      IF (IPRE.EQ.0) THEN
        LI6=LI5+N
        LI7=LI6+N
      ELSE 
        LI7=LI5
      ENDIF
      NIW=LI7-1
      IF (NRW.GT.LRWK) THEN
        IFAIL=NRW-LRWK
        IF (MPRERR.GT.0) WRITE(LUERR,10000) 'real',IFAIL
      ENDIF
      IF (NIW.GT.LIWK) THEN
        IFAIL=NIW-LIWK
        IF (MPRERR.GT.0) WRITE(LUERR,10000) 'integer',IFAIL
10000   FORMAT(A,' ws for preconditioner too small - ',/,
     $         ' more needed: ',I5)
      ENDIF
      IF (IFAIL.NE.0) RETURN
      CALL SLPRE1(N, M, N+1, NELT, A, IA, JA,
     $            NEL, RWK(LR1), IWK(LI1), IWK(LI2),
     $            NU,  RWK(LR2), IWK(LI3), IWK(LI4),
     $            RWK(LR3), IWK(LI5), IWK(LI6), IPRE, LUERR, MPRERR,
     $            IFAIL )
      IF (IPRE.EQ.0 .OR. IPRE.EQ.1) IWK(2)=NEL
      IF (IPRE.EQ.0) IWK(3)=NU
C     end of subroutine slprei
      RETURN
      END
C
C
      SUBROUTINE SLPRE1(N, M, N1, NELT, A, IA, JA,
     $                  NEL, EL, IEL, JEL,
     $                  NU,  U,  IU,  JU, D, NROW, NCOL, IPRE,
     $                  LUERR, MPRERR, IFAIL)
C*    begin prologue slpre1
      INTEGER N,M,N1,NELT
      DOUBLE PRECISION A(NELT)
      INTEGER IA(NELT),JA(NELT)
      INTEGER NEL, NU
      DOUBLE PRECISION EL(NEL),U(NU)
      INTEGER IEL(NEL), JEL(N1), IU(NU), JU(N1)
      DOUBLE PRECISION D(N)
      INTEGER NROW(N),NCOL(N),IPRE,LUERR,MPRERR,IFAIL
C     ------------------------------------------------------------
C
C*    summary :
C
C     S L P R E 1 - call a builtin preconditioner setup subroutine.
C
C*    input parameters
C     ================
C
C     n          int     number of variables and nonlinear equations
C     m          int     size of diagonal blocks (ipre=4 only) -or-
C                        allowed fill-in for each row using ILU(QMR)
C                        (IPRE=5)
C     n1         int     n+1 (for dimensional purposes)
C     nelt       int     see subroutine slprei
C     a(nelt)    double  see subroutine slprei
C     ia(nelt)   int     see subroutine slprei
C     ja(nelt)   int     see subroutine slprei
C     nel        int     size of lower triangle matrix
C                        (nonzeros in slap row format) 
C     nu         int     size of upper triangle matrix
C                        (nonzeros in slap column format) 
C     ipre       int     see iwk(1) description in subroutine slprei
C     luerr      int     see iopt(15) in driver program
C     mprerr     int     see iopt(16) in driver program
C
C*    output parameters
C     =================
C
C     el(nel)    double  real values of lower triangle in slap column-
C                        or values of lower ilu decomposition part in
C                        slap row format
C     iel(nel)   int     row indices of lower triangle or
C                        column indices of lower ilu decomposition part
C     jel(n1)    int     column pointers of lower triangle or
C                        row pointers of lower ilu decomposition part
C     u(nu)      double  real values of upper triangle in slap column
C                        format
C     iu(nu)     int     row indices of upper ilu decomposition part
C     ju(n1)     int     column pointers of upper ilu decomposition part
C     d(n)       double  d(i) holds the real value of the diagonal
C                        scaling or diagonal element of th ilu decomp. 
C     ifail      int     see subroutine slprei
C
C*    workspace parameters
C     ====================
C
C     nrow(n)    int     gets the row permutations during ilu decomp.
C     ncol(n)    int     gets the column permutations during ilu decomp.
C
C     ------------------------------------------------------------
C*    end prologue
      INTEGER NELK, NUK
C
      NELK = NEL
      NUK  = NU
      IF (IPRE.EQ.4) THEN
C       dsbdsc: block diagonal preconditioning set up (by zib)
        CALL DSBDSC(N, NELT, IA, JA, A, 0, M, EL, IEL, IFAIL)
      ELSE IF (IPRE.EQ.2) THEN
C       dsds: diagonal scaling preconditioner slap set up.
        CALL DSDS(N, NELT, IA, JA, A, 0, D)
      ELSE IF (IPRE.EQ.1) THEN
C       ds2lt: lower triangle preconditioner slap set up.
        CALL DS2LT( N, NELT, IA, JA, A, 0, NEL, IEL, JEL, EL )
        NEL = JEL(N+1)-1
      ELSE IF (IPRE.EQ.0) THEN
C       dsilus: incomplete lu decomposition preconditioner slap set up.
        CALL DSILUS(N, NELT, IA, JA, A, 0, NEL, JEL, IEL,
     $     EL, D, NU, IU, JU, U, NROW, NCOL)
        NEL = JEL(N+1)-1
        NU = JU(N+1)-1
      ENDIF
      IF (NEL.GT.NELK) THEN
        IFAIL = NEL-NELK
        IF (MPRERR.GT.0) WRITE(LUERR,10001) 'lower', NEL-NELK
      ENDIF
      IF (NU.GT.NUK) THEN
        IFAIL = IFAIL + NU-NUK
        IF (MPRERR.GT.0) WRITE(LUERR,10001) 'upper', NU-NUK
10001   FORMAT(' SLPREI - real ws. for ',A,' triangle too small:',/,
     $         ' more needed at least : ',I8) 
      ENDIF
      IF (IFAIL.NE.0) IFAIL=-ABS(IFAIL)
C     end of subroutine slpre1
      RETURN
      END
C
      SUBROUTINE SLPREA( N, NELT, A, IA, JA, NEL, NU )
C*    begin prologue slprea
      INTEGER N,NELT
      DOUBLE PRECISION A(NELT)
      INTEGER IA(NELT),JA(NELT)
      INTEGER NEL, NU
C     ------------------------------------------------------------
C
C*    summary :
C
C     S L P R E A - determine the number of elements of a slap
C                   column format matrix, which belong to the lower
C                   and to the upper triangle of the matrix
C
C*    input parameters
C     ================
C
C     n          int     number of variables and nonlinear equations
C     nelt       int     number of nonzeros of the jacobian
C                        stored in slap format in the arrays a, ia, ja
C     a(nelt)    double  jacobian nonzero real values
C     ia(nelt)   int     the corresponding row indices to the real
C                        values stored in a
C     ja(nelt)   int     the column pointers (slap column format)
C
C*    output parameters
C     =================
C
C     nel        int    number of elements in the lower triangle of
C                       a, ia, ja (exclusive the main diagonal)
C     nu         int    number of elements in the upper triangle of
C                       a, ia, ja (exclusive the main diagonal)
C
C     ------------------------------------------------------------
C*    end prologue
C
      INTEGER I, J, IBGN, IEND, IR
C
      NEL = 0
      NU  = 0
      DO 10 J=1,N
        IBGN = JA(J)
        IEND = JA(J+1)-1
        DO 11 I=IBGN,IEND
          IR = IA(I)
          IF (IR.LT.J) THEN 
            NU = NU+1
          ELSE IF (IR.GT.J) THEN 
            NEL = NEL+1
          ENDIF
11      CONTINUE
10    CONTINUE
C     end of subroutine slprea
      RETURN
      END
C
      SUBROUTINE PRECON( N, B, X, RWK, IWK )
C*    begin prologue precon
      INTEGER N
      DOUBLE PRECISION B(N), X(N)
      DOUBLE PRECISION RWK(*)
      INTEGER IWK(*)
C     ------------------------------------------------------------
C
C*    summary :
C
C     P R E C O N - call a preconditioner subroutine from an
C                   iterative solver by ZIB (GBIT, PGBIT).
C                   workspace management routine - the actual call
C                   of the preconditioner is done by subroutine
C                   SLPRE2
C
C*    input parameters
C     ================
C
C         n        int    the number of vector components
C         b(n)     double the right hand side of the system (input)
C         x(n)     double the array to get the solution vector (output)
C         rwk(*)   double user workspace which holds necessary precon-
C                         ditioning information and /or workspace 
C                         to precon. 
C         iwk(*)   int    user workspace (same purpose as rwk(*))
C
C*    output parameters
C     =================
C
C         x(n)     double  the solution of the preconditioning system
C                          m*x=b
C
C     ------------------------------------------------------------
C*    end prologue
C
      INTEGER NEL, NU
      EXTERNAL SLPRE2
C
      IPRE=IWK(9)
      LOWI = 11
      NELT = IWK(6)
      NZMAX = IWK(5)
      NUSRWK = IWK(3)
      LOWIND = LOWI+IWK(4)
      LOWRP = NUSRWK+NZMAX+1
      LOWIP = LOWIND+NZMAX+N+1
      IF (IPRE.EQ.0 .OR. IPRE.EQ.1)
     $    NEL=IWK(LOWIP+1)
      IF (IPRE.EQ.0) NU = IWK(LOWIP+2)
      IF (IPRE.EQ.4) THEN
        M=IWK(10)
        NEL=N*M
      ENDIF
      LR1=LOWRP
      IF ( IPRE.EQ.0 .OR. IPRE.EQ.1 .OR. IPRE.EQ.4 ) THEN
        LR2=LR1+NEL
      ELSE
        LR2=LR1
      ENDIF
      IF (IPRE.EQ.0) THEN
        LR3=LR2+NU
      ELSE
        LR3=LR2
      ENDIF
      LI1=LOWIP+1
      IF (IPRE.EQ.0) LI1=LOWIP+3
      IF (IPRE.EQ.1 .OR. IPRE.EQ.4)
     $     LI1=LOWIP+2
      IF (IPRE.EQ.0 .OR. IPRE.EQ.1) THEN
        LI2=LI1+NEL
        LI3=LI2+N+1
      ELSE IF (IPRE.EQ.4) THEN
        LI2=LI1+NEL
        LI3=LI2
      ELSE
        LI2=LI1
        LI3=LI1
      ENDIF
      IF (IPRE.EQ.0) THEN
        LI4=LI3+NU
      ELSE
        LI4=LI3
        LI5=LI4
      ENDIF
      CALL SLPRE2(N, M, N+1, NELT, 
     $            RWK(NUSRWK+1), IWK(LOWIND), IWK(LOWIND+NZMAX),
     $            B, X,
     $            NEL, RWK(LR1), IWK(LI1), IWK(LI2),
     $            NU,  RWK(LR2), IWK(LI3), IWK(LI4),
     $            RWK(LR3), IWK(LI5), IPRE)
C     end of subroutine precon
      RETURN
      END
C
      SUBROUTINE SLPRE2(N, M, N1, NELT, A, IA, JA, B, X,
     $                  NEL, EL, IEL, JEL, NU, U, IU, JU, D,
     $                  IDA, IPRE)
C*    begin prologue slpre2
      INTEGER N,M,N1,NELT
      DOUBLE PRECISION A(NELT)
      INTEGER IA(NELT),JA(NELT)
      DOUBLE PRECISION B(N),X(N)
      DOUBLE PRECISION EL(NEL),U(NU)
      INTEGER IEL(NEL), JEL(N1), IU(NU), JU(N1)
      DOUBLE PRECISION D(N)
      INTEGER IDA(N), IPRE
C     ------------------------------------------------------------
C
C*    summary :
C
C     s l p r e 2 - call a builtin preconditioner subroutine.
C
C*    input parameters
C     ================
C
C     n, m, nelt, a(nelt), ia(nelt), ja(nelt) :
C     see parameter description of subroutine slprec
C
C     n1, nel, el(nel), iel(nel), jel(n1),
C     nu, u(nu), iu(nu), ju(n1), d(n), ipre :
C     see parameter description of preconditioner setup subroutine 
C     slpre1
C
C     b(n)       double  the right hand side of the preconditioning
C                        system m*x=b .
C
C*    output parameters
C     =================
C
C     x(n)       double  the solution of the preconditioning system
C
C     ------------------------------------------------------------
C*    end prologue
      INTEGER IW(10), IDUM
      DOUBLE PRECISION DUM
C
      IF (IPRE.EQ.4) THEN
C       dsbdic: block diagonal preconditioning (by zib)
        CALL DSBDIC(N, B, X, NELT, IA, JA, A, 0, M, EL, IEL)
      ELSE IF (IPRE.EQ.3) THEN
        DO 10 I=1,N
          X(I) = B(I)
10      CONTINUE
      ELSE IF (IPRE.EQ.2) THEN
        IW(4)=1
C       dsdi: diagonal matrix vector multiply.
        CALL DSDI(N, B, X, IDUM, IDUM, IDUM, DUM, 0, D, IW)
      ELSE IF (IPRE.EQ.1) THEN
C       dsli2: lower triangle matrix backsolve.
        CALL DSLI2(N, B, X, NELT, IEL, JEL, EL)
      ELSE IF (IPRE.EQ.0) THEN
C       dslui2: slap back solve for ldu factorization.
        CALL DSLUI2(N, B, X, JEL, IEL, EL, D, IU, JU, U )
      ENDIF
C     end of subroutine slpre2
      RETURN
      END
C
      SUBROUTINE MULJAC ( N, X, B, RWK, IWK )
C*    begin prologue muljac
      INTEGER N
      DOUBLE PRECISION X(N), B(N)
      DOUBLE PRECISION RWK(*)
      INTEGER IWK(*)
C     ------------------------------------------------------------
C
C*    summary :
C
C     M U L J A C - call a matrix times vector subroutine from
C                   the SLAP package.
C                   workspace management routine - the actual call
C                   of the preconditioner is done by subroutine
C                   SLPRE2
C
C*    input parameters
C     ================
C
C         N        Int    The number of vector components
C         X(N)     Double The vector to be multiplied by the Jacobian
C                         (input)
C         B(N)     Double The array to get the result vector
C                         Jacobian * X (output)
C         RWK(*)   Double User workspace which holds necessary 
C                         information about the matrix and /or workspace 
C                         to MULJAC. 
C         IWK(*)   Int    User workspace (same purpose as RWK(*))
C
C     ------------------------------------------------------------
C*    end prologue
C
      PARAMETER (LOWI=11)
C
      NELT = IWK(6)
      NZMAX = IWK(5)
      NUSRWK = IWK(3)
      LOWIND = LOWI+IWK(4)
      CALL DSMV(N, X, B, NELT, IWK(LOWIND), IWK(LOWIND+NZMAX),
     $          RWK(NUSRWK+1), 0)
      RETURN
      END
      SUBROUTINE DSMV( N, X, Y, NELT, IA, JA, A, ISYM )
C***BEGIN PROLOGUE  DSMV
C***DATE WRITTEN   871119   (YYMMDD)
C***REVISION DATE  881213   (YYMMDD)
C***CATEGORY NO.  D2A4, D2B4
C***KEYWORDS  LIBRARY=SLATEC(SLAP),
C             TYPE=DOUBLE PRECISION(DSMV-S),
C             Matrix Vector Multiply, Sparse
C***AUTHOR  Greenbaum, Anne, Courant Institute
C           Seager, Mark K., (LLNL)
C             Lawrence Livermore National Laboratory
C             PO BOX 808, L-300
C             Livermore, CA 94550 (415) 423-3141
C             seager@lll-crg.llnl.gov
C***PURPOSE  SLAP Column Format Sparse Matrix Vector Product.
C            Routine to calculate the sparse matrix vector product:
C            Y = A*X.
C***DESCRIPTION
C *Usage:
C     INTEGER  N, NELT, IA(NELT), JA(N+1), ISYM
C     DOUBLE PRECISION X(N), Y(N), A(NELT)
C
C     CALL DSMV(N, X, Y, NELT, IA, JA, A, ISYM )
C         
C *Arguments:
C N      :IN       Integer.
C         Order of the Matrix.
C X      :IN       Double Precision X(N).
C         The vector that should be multiplied by the matrix.
C Y      :OUT      Double Precision Y(N).
C         The product of the matrix and the vector.
C NELT   :IN       Integer.
C         Number of Non-Zeros stored in A.
C IA     :IN       Integer IA(NELT).
C JA     :IN       Integer JA(N+1).
C A      :IN       Integer A(NELT).
C         These arrays should hold the matrix A in the SLAP Column
C         format.  See "Description", below. 
C ISYM   :IN       Integer.
C         Flag to indicate symmetric storage format.
C         If ISYM=0, all nonzero entries of the matrix are stored.
C         If ISYM=1, the matrix is symmetric, and only the upper
C         or lower triangle of the matrix is stored.
C
C *Description
C       =================== S L A P Column format ==================
C       This routine requires  that the  matrix  A be  stored in the
C       SLAP Column format.  In this format the non-zeros are stored
C       counting down columns (except for  the diagonal entry, which
C       must appear first in each  "column")  and are stored  in the
C       double precision array A.   In other words,  for each column
C       in the matrix put the diagonal entry in  A.  Then put in the
C       other non-zero  elements going down  the column (except  the
C       diagonal) in order.   The  IA array holds the  row index for
C       each non-zero.  The JA array holds the offsets  into the IA,
C       A arrays  for  the  beginning  of each   column.   That  is,
C       IA(JA(ICOL)),  A(JA(ICOL)) points   to the beginning  of the
C       ICOL-th   column    in    IA and   A.      IA(JA(ICOL+1)-1),
C       A(JA(ICOL+1)-1) points to  the  end of the   ICOL-th column.
C       Note that we always have  JA(N+1) = NELT+1,  where N is  the
C       number of columns in  the matrix and NELT  is the number  of
C       non-zeros in the matrix.
C       
C       Here is an example of the  SLAP Column  storage format for a
C       5x5 Matrix (in the A and IA arrays '|'  denotes the end of a 
C       column):
C       
C       5x5 Matrix      SLAP Column format for 5x5 matrix on left.
C       1  2  3    4  5    6  7    8    9 10 11
C       |11 12  0  0 15|   A: 11 21 51 | 22 12 | 33 53 | 44 | 55 15 35
C       |21 22  0  0  0|  IA:  1  2  5 |  2  1 |  3  5 |  4 |  5  1  3
C       | 0  0 33  0 35|  JA:  1  4  6    8  9   12
C       | 0  0  0 44  0|
C       |51  0 53  0 55|
C       
C       With  the SLAP  format  the "inner  loops" of  this  routine
C       should vectorize   on machines with   hardware  support  for
C       vector gather/scatter operations.  Your compiler may require
C       a  compiler directive  to  convince   it that there  are  no
C       implicit vector  dependencies.  Compiler directives  for the
C       Alliant FX/Fortran and CRI CFT/CFT77 compilers  are supplied
C       with the standard SLAP distribution.
C
C *Precision:           Double Precision
C *Cautions:
C     This   routine   assumes  that  the matrix A is stored in SLAP 
C     Column format.  It does not check  for  this (for  speed)  and 
C     evil, ugly, ornery and nasty things  will happen if the matrix 
C     data  structure  is,  in fact, not SLAP Column.  Beware of the 
C     wrong data structure!!!
C
C *See Also:
C       DSMTV
C***REFERENCES  (NONE)
C***ROUTINES CALLED  (NONE)
C***END PROLOGUE  DSMV
      IMPLICIT DOUBLE PRECISION(A-H,O-Z)
      INTEGER N, NELT, IA(NELT), JA(NELT), ISYM
      DOUBLE PRECISION A(NELT), X(N), Y(N)
C
C         Zero out the result vector.
C***FIRST EXECUTABLE STATEMENT  DSMV
      DO 10 I = 1, N
         Y(I) = 0.0D0
 10   CONTINUE
C
C         Multiply by A.
C
CVD$R NOCONCUR
      DO 30 ICOL = 1, N
         IBGN = JA(ICOL)
         IEND = JA(ICOL+1)-1
CLLL. OPTION ASSERT (NOHAZARD)
CDIR$ IVDEP
CVD$ NODEPCHK
         DO 20 I = IBGN, IEND
            Y(IA(I)) = Y(IA(I)) + A(I)*X(ICOL)
 20      CONTINUE
 30   CONTINUE
C
      IF( ISYM.EQ.1 ) THEN
C
C         The matrix is non-symmetric.  Need to get the other half in...
C         This loops assumes that the diagonal is the first entry in
C         each column.
C
         DO 50 IROW = 1, N
            JBGN = JA(IROW)+1
            JEND = JA(IROW+1)-1
            IF( JBGN.GT.JEND ) GOTO 50
            DO 40 J = JBGN, JEND
               Y(IROW) = Y(IROW) + A(J)*X(IA(J))
 40         CONTINUE
 50      CONTINUE
      ENDIF
      RETURN
C------------- LAST LINE OF DSMV FOLLOWS ----------------------------
      END
      SUBROUTINE DSDS(N, NELT, IA, JA, A, ISYM, DINV)
C***BEGIN PROLOGUE  DSDS
C***DATE WRITTEN   890404   (YYMMDD)
C***REVISION DATE  890404   (YYMMDD)
C***CATEGORY NO.  D2A4, D2B4
C***KEYWORDS  LIBRARY=SLATEC(SLAP),
C             TYPE=DOUBLE PRECISION(DSDS-D),
C             SLAP Sparse, Diagonal
C***AUTHOR  Greenbaum, Anne, Courant Institute
C           Seager, Mark K., (LLNL)
C             Lawrence Livermore National Laboratory
C             PO BOX 808, L-300
C             Livermore, CA 94550 (415) 423-3141
C             seager@lll-crg.llnl.gov
C***PURPOSE  Diagonal Scaling Preconditioner SLAP Set Up.
C            Routine to compute the inverse of the diagonal of a matrix
C            stored in the SLAP Column format.
C***DESCRIPTION
C *Usage:
C     INTEGER N, NELT, IA(NELT), JA(NELT), ISYM
C     DOUBLE PRECISION A(NELT), DINV(N)
C
C     CALL DSDS( N, NELT, IA, JA, A, ISYM, DINV )
C
C *Arguments:
C N      :IN       Integer.
C         Order of the Matrix.
C NELT   :IN       Integer.
C         Number of elements in arrays IA, JA, and A.
C IA     :INOUT    Integer IA(NELT).
C JA     :INOUT    Integer JA(NELT).
C A      :INOUT    Double Precision A(NELT).
C         These arrays should hold the matrix A in the SLAP Column
C         format.  See "Description", below. 
C ISYM   :IN       Integer.
C         Flag to indicate symmetric storage format.
C         If ISYM=0, all nonzero entries of the matrix are stored.
C         If ISYM=1, the matrix is symmetric, and only the upper
C         or lower triangle of the matrix is stored.
C DINV   :OUT      Double Precision DINV(N).
C         Upon return this array holds 1./DIAG(A).
C
C *Description
C       =================== S L A P Column format ==================
C       This routine requires  that the  matrix  A be  stored in the
C       SLAP Column format.  In this format the non-zeros are stored
C       counting down columns (except for  the diagonal entry, which
C       must appear first in each  "column")  and are stored  in the
C       double precision array A.   In other words,  for each column
C       in the matrix put the diagonal entry in  A.  Then put in the
C       other non-zero  elements going down  the column (except  the
C       diagonal) in order.   The  IA array holds the  row index for
C       each non-zero.  The JA array holds the offsets  into the IA,
C       A arrays  for  the  beginning  of each   column.   That  is,
C       IA(JA(ICOL)),  A(JA(ICOL)) points   to the beginning  of the
C       ICOL-th   column    in    IA and   A.      IA(JA(ICOL+1)-1),
C       A(JA(ICOL+1)-1) points to  the  end of the   ICOL-th column.
C       Note that we always have  JA(N+1) = NELT+1,  where N is  the
C       number of columns in  the matrix and NELT  is the number  of
C       non-zeros in the matrix.
C       
C       Here is an example of the  SLAP Column  storage format for a
C       5x5 Matrix (in the A and IA arrays '|'  denotes the end of a 
C       column):
C       
C       5x5 Matrix      SLAP Column format for 5x5 matrix on left.
C       1  2  3    4  5    6  7    8    9 10 11
C       |11 12  0  0 15|   A: 11 21 51 | 22 12 | 33 53 | 44 | 55 15 35
C       |21 22  0  0  0|  IA:  1  2  5 |  2  1 |  3  5 |  4 |  5  1  3
C       | 0  0 33  0 35|  JA:  1  4  6    8  9   12
C       | 0  0  0 44  0|
C       |51  0 53  0 55|
C       
C       With the SLAP  format  all  of  the   "inner  loops" of this
C       routine should vectorize  on  machines with hardware support
C       for vector   gather/scatter  operations.  Your compiler  may
C       require a compiler directive to  convince it that  there are
C       no  implicit  vector  dependencies.  Compiler directives for
C       the Alliant    FX/Fortran and CRI   CFT/CFT77 compilers  are
C       supplied with the standard SLAP distribution.
C
C *Precision:           Double Precision
C
C *Cautions:
C       This routine assumes that the diagonal of A is all  non-zero
C       and that the operation DINV = 1.0/DIAG(A) will not underflow
C       or overflow.    This  is done so that the  loop  vectorizes.
C       Matricies with zero or near zero or very  large entries will
C       have numerical difficulties  and  must  be fixed before this 
C       routine is called.
C***REFERENCES  (NONE)
C***ROUTINES CALLED  (NONE)
C***END PROLOGUE  DSDS
      IMPLICIT DOUBLE PRECISION(A-H,O-Z)
      INTEGER N, NELT, IA(NELT), JA(NELT), ISYM
      DOUBLE PRECISION A(NELT), DINV(N)
C
C         Assume the Diagonal elements are the first in each column.
C         This loop should *VECTORIZE*.  If it does not you may have
C         to add a compiler directive.  We do not check for a zero
C         (or near zero) diagonal element since this would interfere 
C         with vectorization.  If this makes you nervous put a check
C         in!  It will run much slower.
C***FIRST EXECUTABLE STATEMENT  DSDS
 1    CONTINUE
      DO 10 ICOL = 1, N
         DINV(ICOL) = 1.0D0/A(JA(ICOL))
 10   CONTINUE
C         
      RETURN
C------------- LAST LINE OF DSDS FOLLOWS ----------------------------
      END
      SUBROUTINE DSDI(N, B, X, NELT, IA, JA, A, ISYM, RWORK, IWORK)
C***BEGIN PROLOGUE  DSDI
C***DATE WRITTEN   871119   (YYMMDD)
C***REVISION DATE  881213  (YYMMDD)
C***CATEGORY NO.  D2A4, D2B4
C***KEYWORDS  LIBRARY=SLATEC(SLAP),
C             TYPE=DOUBLE PRECISION(DSDI-S),
C             Linear system solve, Sparse, Iterative Precondition
C***AUTHOR  Greenbaum, Anne, Courant Institute
C           Seager, Mark K., (LLNL)
C             Lawrence Livermore National Laboratory
C             PO BOX 808, L-300
C             Livermore, CA 94550 (415) 423-3141
C             seager@lll-crg.llnl.gov
C***PURPOSE  Diagonal Matrix Vector Multiply.
C            Routine to calculate the product  X = DIAG*B,  
C            where DIAG is a diagonal matrix.
C***DESCRIPTION
C *Usage:
C *Arguments:
C N      :IN       Integer
C         Order of the Matrix.
C B      :IN       Double Precision B(N).
C         Vector to multiply the diagonal by.
C X      :OUT      Double Precision X(N).
C         Result of DIAG*B.
C NELT   :DUMMY    Integer.
C         Retained for compatibility with SLAP MSOLVE calling sequence.
C IA     :DUMMY    Integer IA(NELT).
C         Retained for compatibility with SLAP MSOLVE calling sequence.
C JA     :DUMMY    Integer JA(N+1).
C         Retained for compatibility with SLAP MSOLVE calling sequence.
C  A     :DUMMY    Double Precision A(NELT).
C         Retained for compatibility with SLAP MSOLVE calling sequence.
C ISYM   :DUMMY    Integer.
C         Retained for compatibility with SLAP MSOLVE calling sequence.
C RWORK  :IN       Double Precision RWORK(USER DEFINABLE).
C         Work array holding the diagonal of some matrix to scale
C         B by.  This array must be set by the user or by a call
C         to the slap routine DSDS or DSD2S.  The length of RWORK
C         must be > IWORK(4)+N.
C IWORK  :IN       Integer IWORK(10).
C         IWORK(4) holds the offset into RWORK for the diagonal matrix
C         to scale B by.  This is usually set up by the SLAP pre-
C         conditioner setup routines DSDS or DSD2S.
C
C *Description:
C         This routine is supplied with the SLAP package to perform
C         the  MSOLVE  operation for iterative drivers that require
C         diagonal  Scaling  (e.g., DSDCG, DSDBCG).   It  conforms
C         to the SLAP MSOLVE CALLING CONVENTION  and hence does not
C         require an interface routine as do some of the other pre-
C         conditioners supplied with SLAP.
C
C *Precision:           Double Precision
C *See Also:
C       DSDS, DSD2S
C***REFERENCES  (NONE)
C***ROUTINES CALLED  (NONE)
C***END PROLOGUE  DSDI
      IMPLICIT DOUBLE PRECISION(A-H,O-Z)
      INTEGER N, NELT, IA(NELT), JA(NELT), ISYM, IWORK(10)
      DOUBLE PRECISION B(N), X(N), A(NELT), RWORK(1)
C
C         Determine where the inverse of the diagonal 
C         is in the work array and then scale by it.
C***FIRST EXECUTABLE STATEMENT  DSDI
      LOCD = IWORK(4) - 1
      DO 10 I = 1, N
         X(I) = RWORK(LOCD+I)*B(I)
 10   CONTINUE
      RETURN
C------------- LAST LINE OF DSDI FOLLOWS ----------------------------
      END
      SUBROUTINE DS2LT( N, NELT, IA, JA, A, ISYM, NEL, IEL, JEL, EL )
C***BEGIN PROLOGUE  DS2LT
C***DATE WRITTEN   890404   (YYMMDD)
C***REVISION DATE  890404   (YYMMDD)
C***CATEGORY NO.  D2A4, D2B4
C***KEYWORDS  LIBRARY=SLATEC(SLAP),
C             TYPE=DOUBLE PRECISION(DS2LT-D),
C             Linear system, SLAP Sparse, Lower Triangle
C***AUTHOR  Greenbaum, Anne, Courant Institute
C           Seager, Mark K., (LLNL)
C             Lawrence Livermore National Laboratory
C             PO BOX 808, L-300
C             Livermore, CA 94550 (415) 423-3141
C             seager@lll-crg.llnl.gov
C***PURPOSE  Lower Triangle Preconditioner SLAP Set Up.
C            Routine to store the lower triangle of a matrix stored
C            in the Slap Column format.
C***DESCRIPTION
C *Usage:
C     INTEGER N, NELT, IA(NELT), JA(NELT), ISYM
C     INTEGER NEL, IEL(N+1), JEL(NEL), NROW(N)
C     DOUBLE PRECISION A(NELT), EL(NEL)
C
C     CALL DS2LT( N, NELT, IA, JA, A, ISYM, NEL, IEL, JEL, EL )
C
C *Arguments:
C N      :IN       Integer
C         Order of the Matrix.
C NELT   :IN       Integer.
C         Number of non-zeros stored in A.
C IA     :IN       Integer IA(NELT).
C JA     :IN       Integer JA(NELT).
C A      :IN       Double Precision A(NELT).
C         These arrays should hold the matrix A in the SLAP Column
C         format.  See "Description", below. 
C ISYM   :IN       Integer.
C         Flag to indicate symmetric storage format.
C         If ISYM=0, all nonzero entries of the matrix are stored.
C         If ISYM=1, the matrix is symmetric, and only the lower
C         triangle of the matrix is stored.
C NEL    :OUT      Integer.
C         Number of non-zeros in the lower triangle of A.   Also 
C         coresponds to the length of the JEL, EL arrays.
C IEL    :OUT      Integer IEL(N+1).
C JEL    :OUT      Integer JEL(NEL).
C EL     :OUT      Double Precision     EL(NEL).
C         IEL, JEL, EL contain the lower triangle of the A matrix
C         stored in SLAP Column format.  See "Description", below
C         for more details bout the SLAP Column format.
C
C *Description
C       =================== S L A P Column format ==================
C       This routine requires  that the  matrix  A be  stored in the
C       SLAP Column format.  In this format the non-zeros are stored
C       counting down columns (except for  the diagonal entry, which
C       must appear first in each  "column")  and are stored  in the
C       double precision array A.   In other words,  for each column
C       in the matrix put the diagonal entry in  A.  Then put in the
C       other non-zero  elements going down  the column (except  the
C       diagonal) in order.   The  IA array holds the  row index for
C       each non-zero.  The JA array holds the offsets  into the IA,
C       A arrays  for  the  beginning  of each   column.   That  is,
C       IA(JA(ICOL)),  A(JA(ICOL)) points   to the beginning  of the
C       ICOL-th   column    in    IA and   A.      IA(JA(ICOL+1)-1),
C       A(JA(ICOL+1)-1) points to  the  end of the   ICOL-th column.
C       Note that we always have  JA(N+1) = NELT+1,  where N is  the
C       number of columns in  the matrix and NELT  is the number  of
C       non-zeros in the matrix.
C       
C       Here is an example of the  SLAP Column  storage format for a
C       5x5 Matrix (in the A and IA arrays '|'  denotes the end of a 
C       column):
C       
C       5x5 Matrix      SLAP Column format for 5x5 matrix on left.
C       1  2  3    4  5    6  7    8    9 10 11
C       |11 12  0  0 15|   A: 11 21 51 | 22 12 | 33 53 | 44 | 55 15 35
C       |21 22  0  0  0|  IA:  1  2  5 |  2  1 |  3  5 |  4 |  5  1  3
C       | 0  0 33  0 35|  JA:  1  4  6    8  9   12
C       | 0  0  0 44  0|
C       |51  0 53  0 55|
C       
C *Precision:           Double Precision
C***REFERENCES  (NONE)
C***ROUTINES CALLED  (NONE)
C***END PROLOGUE  DS2LT
      IMPLICIT DOUBLE PRECISION(A-H,O-Z)
      INTEGER N, NELT, IA(NELT), JA(NELT), ISYM 
      INTEGER NEL, IEL(NEL), JEL(NEL)
      DOUBLE PRECISION A(NELT), EL(NELT)
C***FIRST EXECUTABLE STATEMENT  DS2LT
      IF( ISYM.EQ.0 ) THEN
C
C         The matrix is stored non-symmetricly.  Pick out the lower
C         triangle.
C
         NEL = 0
         DO 20 ICOL = 1, N
            JEL(ICOL) = NEL+1
            JBGN = JA(ICOL)
            JEND = JA(ICOL+1)-1
CVD$ NOVECTOR
            DO 10 J = JBGN, JEND
               IF( IA(J).GE.ICOL ) THEN
                  NEL = NEL + 1
                  IEL(NEL) = IA(J)
                  EL(NEL)  = A(J)
               ENDIF
 10         CONTINUE
 20      CONTINUE
         JEL(N+1) = NEL+1
      ELSE
C
C         The matrix is symmetric and only the lower triangle is 
C         stored.  Copy it to IEL, JEL, EL.
C
         NEL = NELT
         DO 30 I = 1, NELT
            IEL(I) = IA(I)
            EL(I) = A(I)
 30      CONTINUE
         DO 40 I = 1, N+1
            JEL(I) = JA(I)
 40      CONTINUE
      ENDIF
      RETURN
C------------- LAST LINE OF DS2LT FOLLOWS ----------------------------
      END
      SUBROUTINE DSLI2(N, B, X, NEL, IEL, JEL, EL)
C***BEGIN PROLOGUE  DSLI2
C***DATE WRITTEN   871119   (YYMMDD)
C***REVISION DATE  881213   (YYMMDD)
C***CATEGORY NO.  D2A4
C***KEYWORDS  LIBRARY=SLATEC(SLAP),
C             TYPE=DOUBLE PRECISION(DSLI2-S),
C             Linear system solve, Sparse, Iterative Precondition
C***AUTHOR  Greenbaum, Anne, Courant Institute
C           Seager, Mark K., (LLNL)
C             Lawrence Livermore National Laboratory
C             PO BOX 808, L-300
C             Livermore, CA 94550 (415) 423-3141
C             seager@lll-crg.llnl.gov
C***PURPOSE  SLAP for Lower Triangle Matrix Backsolve.
C            Routine to solve a system of the form  Lx = b , where
C            L is a lower triangular matrix.
C***DESCRIPTION
C *Usage:
C     INTEGER N,  NEL, IEL(N+1), JEL(NEL)
C     DOUBLE PRECISION B(N), X(N), EL(NEL)
C
C     CALL DSLI2( N, B, X, NEL, IEL, JEL, EL )
C
C *Arguments:
C N      :IN       Integer
C         Order of the Matrix.
C B      :IN       Double Precision B(N).
C         Right hand side vector.
C X      :OUT      Double Precision X(N).
C         Solution to Lx = b.
C NEL    :IN       Integer.
C         Number of non-zeros in the EL array.
C IEL    :IN       Integer IEL(N+1).
C JEL    :IN       Integer JEL(NEL).
C EL     :IN       Double Precision EL(NEL).
C         IEL, JEL, EL contain the unit lower triangular factor   of
C         the incomplete decomposition   of the A  matrix  stored in 
C         SLAP Row format.  The diagonal of  ones *IS* stored.  This 
C         structure can be set up by the  DS2LT  routine.  See "LONG 
C         DESCRIPTION", below for more details about  the  SLAP  Row 
C         format.
C
C *Description:
C       This routine is supplied with the SLAP package  as a routine
C       to  perform the  MSOLVE operation in  the SIR for the driver
C       routine DSGS.  It must be called via the SLAP MSOLVE calling
C       sequence convention interface routine DSLI.
C         **** THIS ROUTINE ITSELF DOES NOT CONFORM TO THE ****
C               **** SLAP MSOLVE CALLING CONVENTION ****
C
C       ==================== S L A P Row format ====================
C       This routine requires  that the matrix A  be  stored  in the
C       SLAP  Row format.   In this format  the non-zeros are stored
C       counting across  rows (except for the diagonal  entry, which
C       must appear first in each "row") and  are stored in the 
C       double precision
C       array A.  In other words, for each row in the matrix put the
C       diagonal entry in  A.   Then   put  in the   other  non-zero
C       elements   going  across the  row (except   the diagonal) in
C       order.   The  JA array  holds   the column   index for  each
C       non-zero.   The IA  array holds the  offsets into  the JA, A
C       arrays  for   the   beginning  of   each  row.   That    is,
C       JA(IA(IROW)),  A(IA(IROW)) points  to  the beginning  of the
C       IROW-th row in JA and A.   JA(IA(IROW+1)-1), A(IA(IROW+1)-1)
C       points to the  end of the  IROW-th row.  Note that we always
C       have IA(N+1) =  NELT+1, where  N  is  the number of rows  in
C       the matrix  and NELT  is the  number   of  non-zeros in  the
C       matrix.
C       
C       Here is an example of the SLAP Row storage format for a  5x5
C       Matrix (in the A and JA arrays '|' denotes the end of a row):
C
C           5x5 Matrix         SLAP Row format for 5x5 matrix on left.
C                              1  2  3    4  5    6  7    8    9 10 11
C       |11 12  0  0 15|   A: 11 12 15 | 22 21 | 33 35 | 44 | 55 51 53
C       |21 22  0  0  0|  JA:  1  2  5 |  2  1 |  3  5 |  4 |  5  1  3
C       | 0  0 33  0 35|  IA:  1  4  6    8  9   12
C       | 0  0  0 44  0|  
C       |51  0 53  0 55|  
C
C       With  the SLAP  Row format  the "inner loop" of this routine
C       should vectorize   on machines with   hardware  support  for
C       vector gather/scatter operations.  Your compiler may require
C       a  compiler directive  to  convince   it that there  are  no
C       implicit vector  dependencies.  Compiler directives  for the
C       Alliant FX/Fortran and CRI CFT/CFT77 compilers  are supplied
C       with the standard SLAP distribution.
C
C *Precision: Double Precision
C *See Also:
C         DSLI
C***REFERENCES  (NONE)
C***ROUTINES CALLED  (NONE)
C***END PROLOGUE  DSLI2
      IMPLICIT DOUBLE PRECISION(A-H,O-Z)
      INTEGER N, NEL, IEL(NEL), JEL(NEL)
      DOUBLE PRECISION B(N), X(N), EL(NEL)
C
C         Initialize the solution by copying the right hands side
C         into it.
C***FIRST EXECUTABLE STATEMENT  DSLI2
      DO 10 I=1,N
         X(I) = B(I)
 10   CONTINUE
C         
CVD$ NOCONCUR
      DO 30 ICOL = 1, N
         X(ICOL) = X(ICOL)/EL(JEL(ICOL))
         JBGN = JEL(ICOL) + 1
         JEND = JEL(ICOL+1) - 1
         IF( JBGN.LE.JEND ) THEN
CLLL. OPTION ASSERT (NOHAZARD)
CDIR$ IVDEP
CVD$ NOCONCUR
CVD$ NODEPCHK
            DO 20 J = JBGN, JEND
               X(IEL(J)) = X(IEL(J)) - EL(J)*X(ICOL)
 20         CONTINUE
         ENDIF
 30   CONTINUE
C         
      RETURN
C------------- LAST LINE OF DSLI2 FOLLOWS ----------------------------
      END
      SUBROUTINE DSILUS(N, NELT, IA, JA, A, ISYM, NL, IL, JL,
     $     L, DINV, NU, IU, JU, U, NROW, NCOL)
C***BEGIN PROLOGUE  DSILUS
C***DATE WRITTEN   890404   (YYMMDD)
C***REVISION DATE  890404   (YYMMDD)
C***CATEGORY NO.  D2A4, D2B4
C***KEYWORDS  LIBRARY=SLATEC(SLAP),
C             TYPE=DOUBLE PRECISION(DSILUS-D),
C             Non-Symmetric Linear system, Sparse, 
C             Iterative Precondition, Incomplete LU Factorization
C***AUTHOR  Greenbaum, Anne, Courant Institute
C           Seager, Mark K., (LLNL)
C             Lawrence Livermore National Laboratory
C             PO BOX 808, L-300
C             Livermore, CA 94550 (415) 423-3141
C             seager@lll-crg.llnl.gov
C***PURPOSE  Incomplete LU Decomposition Preconditioner SLAP Set Up.
C            Routine to generate the incomplete LDU decomposition of a 
C            matrix.  The  unit lower triangular factor L is stored by 
C            rows and the  unit upper triangular factor U is stored by 
C            columns.  The inverse of the diagonal matrix D is stored.
C            No fill in is allowed.
C***DESCRIPTION
C *Usage:
C     INTEGER N, NELT, IA(NELT), JA(NELT), ISYM
C     INTEGER NL, IL(N+1), JL(NL), NU, IU(N+1), JU(NU)
C     INTEGER NROW(N), NCOL(N)
C     DOUBLE PRECISION A(NELT), L(NL), U(NU), DINV(N)
C
C     CALL DSILUS( N, NELT, IA, JA, A, ISYM, NL, IL, JL, L, 
C    $    DINV, NU, IU, JU, U, NROW, NCOL )
C
C *Arguments:
C N      :IN       Integer
C         Order of the Matrix.
C NELT   :IN       Integer.
C         Number of elements in arrays IA, JA, and A.
C IA     :IN       Integer IA(NELT).
C JA     :IN       Integer JA(NELT).
C A      :IN       Double Precision A(NELT).
C         These arrays should hold the matrix A in the SLAP Column
C         format.  See "Description", below. 
C ISYM   :IN       Integer.
C         Flag to indicate symmetric storage format.
C         If ISYM=0, all nonzero entries of the matrix are stored.
C         If ISYM=1, the matrix is symmetric, and only the lower 
C         triangle of the matrix is stored.
C NL     :OUT      Integer.
C         Number of non-zeros in the EL array.
C IL     :OUT      Integer IL(N+1).
C JL     :OUT      Integer JL(NL).
C L      :OUT      Double Precision L(NL).
C         IL, JL, L  contain the unit ower  triangular factor of  the
C         incomplete decomposition  of some  matrix stored  in   SLAP
C         Row format.     The   Diagonal  of ones  *IS*  stored.  See
C         "DESCRIPTION", below for more details about the SLAP format.
C NU     :OUT      Integer.
C         Number of non-zeros in the U array.     
C IU     :OUT      Integer IU(N+1).
C JU     :OUT      Integer JU(NU).
C U      :OUT      Double Precision     U(NU).
C         IU, JU, U contain   the unit upper triangular factor of the
C         incomplete  decomposition    of some matrix  stored in SLAP
C         Column  format.   The Diagonal of ones   *IS*  stored.  See 
C         "Description", below  for  more  details  about  the   SLAP 
C         format.
C NROW   :WORK     Integer NROW(N).
C         NROW(I) is the number of non-zero elements in the I-th row
C         of L.
C NCOL   :WORK     Integer NCOL(N).
C         NCOL(I) is the number of non-zero elements in the I-th 
C         column of U.
C
C *Description
C       IL, JL, L should contain the unit  lower triangular factor of
C       the incomplete decomposition of the A matrix  stored in SLAP
C       Row format.  IU, JU, U should contain  the unit upper factor
C       of the  incomplete decomposition of  the A matrix  stored in
C       SLAP Column format This ILU factorization can be computed by
C       the DSILUS routine.  The diagonals (which is all one's) are
C       stored.
C
C       =================== S L A P Column format ==================
C       This routine requires  that the  matrix  A be  stored in the
C       SLAP Column format.  In this format the non-zeros are stored
C       counting down columns (except for  the diagonal entry, which
C       must appear first in each  "column")  and are stored  in the
C       double precision array A.   In other words,  for each column
C       in the matrix put the diagonal entry in  A.  Then put in the
C       other non-zero  elements going down  the column (except  the
C       diagonal) in order.   The  IA array holds the  row index for
C       each non-zero.  The JA array holds the offsets  into the IA,
C       A arrays  for  the  beginning  of each   column.   That  is,
C       IA(JA(ICOL)),  A(JA(ICOL)) points   to the beginning  of the
C       ICOL-th   column    in    IA and   A.      IA(JA(ICOL+1)-1),
C       A(JA(ICOL+1)-1) points to  the  end of the   ICOL-th column.
C       Note that we always have  JA(N+1) = NELT+1,  where N is  the
C       number of columns in  the matrix and NELT  is the number  of
C       non-zeros in the matrix.
C       
C       Here is an example of the  SLAP Column  storage format for a
C       5x5 Matrix (in the A and IA arrays '|'  denotes the end of a 
C       column):
C       
C       5x5 Matrix      SLAP Column format for 5x5 matrix on left.
C       1  2  3    4  5    6  7    8    9 10 11
C       |11 12  0  0 15|   A: 11 21 51 | 22 12 | 33 53 | 44 | 55 15 35
C       |21 22  0  0  0|  IA:  1  2  5 |  2  1 |  3  5 |  4 |  5  1  3
C       | 0  0 33  0 35|  JA:  1  4  6    8  9   12
C       | 0  0  0 44  0|
C       |51  0 53  0 55|
C       
C       ==================== S L A P Row format ====================
C       This routine requires  that the matrix A  be  stored  in the
C       SLAP  Row format.   In this format  the non-zeros are stored
C       counting across  rows (except for the diagonal  entry, which
C       must appear first in each "row") and  are stored in the 
C       double precision
C       array A.  In other words, for each row in the matrix put the
C       diagonal entry in  A.   Then   put  in the   other  non-zero
C       elements   going  across the  row (except   the diagonal) in
C       order.   The  JA array  holds   the column   index for  each
C       non-zero.   The IA  array holds the  offsets into  the JA, A
C       arrays  for   the   beginning  of   each  row.   That    is,
C       JA(IA(IROW)),  A(IA(IROW)) points  to  the beginning  of the
C       IROW-th row in JA and A.   JA(IA(IROW+1)-1), A(IA(IROW+1)-1)
C       points to the  end of the  IROW-th row.  Note that we always
C       have IA(N+1) =  NELT+1, where  N  is  the number of rows  in
C       the matrix  and NELT  is the  number   of  non-zeros in  the
C       matrix.
C       
C       Here is an example of the SLAP Row storage format for a  5x5
C       Matrix (in the A and JA arrays '|' denotes the end of a row):
C
C           5x5 Matrix         SLAP Row format for 5x5 matrix on left.
C                              1  2  3    4  5    6  7    8    9 10 11
C       |11 12  0  0 15|   A: 11 12 15 | 22 21 | 33 35 | 44 | 55 51 53
C       |21 22  0  0  0|  JA:  1  2  5 |  2  1 |  3  5 |  4 |  5  1  3
C       | 0  0 33  0 35|  IA:  1  4  6    8  9   12
C       | 0  0  0 44  0|  
C       |51  0 53  0 55|  
C
C *Precision:           Double Precision
C *See Also:
C       SILUR
C***REFERENCES  1. Gene Golub & Charles Van Loan, "Matrix Computations",
C                 John Hopkins University Press; 3 (1983) IBSN 
C                 0-8018-3010-9.
C***ROUTINES CALLED  (NONE)
C***END PROLOGUE  DSILUS
      IMPLICIT DOUBLE PRECISION(A-H,O-Z)
      INTEGER N, NELT, IA(NELT), JA(NELT), ISYM, NL, IL(NL), JL(NL)
      INTEGER NU, IU(NU), JU(NU), NROW(N), NCOL(N)
      DOUBLE PRECISION A(NELT), L(NL), DINV(N), U(NU)
C         
C         Count number of elements in each row of the lower triangle.
C***FIRST EXECUTABLE STATEMENT  DSILUS
      DO 10 I=1,N
         NROW(I) = 0
         NCOL(I) = 0
 10   CONTINUE
CVD$R NOCONCUR
CVD$R NOVECTOR
      DO 30 ICOL = 1, N
         JBGN = JA(ICOL)+1
         JEND = JA(ICOL+1)-1
         IF( JBGN.LE.JEND ) THEN
            DO 20 J = JBGN, JEND
               IF( IA(J).LT.ICOL ) THEN
                  NCOL(ICOL) = NCOL(ICOL) + 1
               ELSE
                  NROW(IA(J)) = NROW(IA(J)) + 1
                  IF( ISYM.NE.0 ) NCOL(IA(J)) = NCOL(IA(J)) + 1
               ENDIF
 20         CONTINUE
         ENDIF
 30   CONTINUE
      JU(1) = 1
      IL(1) = 1
      DO 40 ICOL = 1, N
         IL(ICOL+1) = IL(ICOL) + NROW(ICOL)
         JU(ICOL+1) = JU(ICOL) + NCOL(ICOL)
         NROW(ICOL) = IL(ICOL)
         NCOL(ICOL) = JU(ICOL)
 40   CONTINUE
C         
C         Copy the matrix A into the L and U structures.
      DO 60 ICOL = 1, N
         DINV(ICOL) = A(JA(ICOL))
         JBGN = JA(ICOL)+1
         JEND = JA(ICOL+1)-1
         IF( JBGN.LE.JEND ) THEN
            DO 50 J = JBGN, JEND
               IROW = IA(J)
               IF( IROW.LT.ICOL ) THEN
C         Part of the upper triangle.
                  IU(NCOL(ICOL)) = IROW
                  U(NCOL(ICOL)) = A(J)
                  NCOL(ICOL) = NCOL(ICOL) + 1
               ELSE
C         Part of the lower triangle (stored by row).
                  JL(NROW(IROW)) = ICOL
                  L(NROW(IROW)) = A(J)
                  NROW(IROW) = NROW(IROW) + 1
                  IF( ISYM.NE.0 ) THEN
C         Symmetric...Copy lower triangle into upper triangle as well.
                     IU(NCOL(IROW)) = ICOL
                     U(NCOL(IROW)) = A(J)
                     NCOL(IROW) = NCOL(IROW) + 1
                  ENDIF
               ENDIF
 50         CONTINUE
         ENDIF
 60   CONTINUE
C
C         Sort the rows of L and the columns of U.
      DO 110 K = 2, N
         JBGN = JU(K)
         JEND = JU(K+1)-1
         IF( JBGN.LT.JEND ) THEN
            DO 80 J = JBGN, JEND-1
               DO 70 I = J+1, JEND
                  IF( IU(J).GT.IU(I) ) THEN
                     ITEMP = IU(J)
                     IU(J) = IU(I)
                     IU(I) = ITEMP
                     TEMP = U(J)
                     U(J) = U(I)
                     U(I) = TEMP
                  ENDIF
 70            CONTINUE
 80         CONTINUE
         ENDIF
         IBGN = IL(K)
         IEND = IL(K+1)-1
         IF( IBGN.LT.IEND ) THEN
            DO 100 I = IBGN, IEND-1
               DO 90 J = I+1, IEND
                  IF( JL(I).GT.JL(J) ) THEN
                     JTEMP = JU(I)
                     JU(I) = JU(J)
                     JU(J) = JTEMP
                     TEMP = L(I)
                     L(I) = L(J)
                     L(J) = TEMP
                  ENDIF
 90            CONTINUE
 100        CONTINUE
         ENDIF
 110  CONTINUE
C
C         Perform the incomplete LDU decomposition.
      DO 300 I=2,N
C         
C           I-th row of L
         INDX1 = IL(I)
         INDX2 = IL(I+1) - 1
         IF(INDX1 .GT. INDX2) GO TO 200
         DO 190 INDX=INDX1,INDX2
            IF(INDX .EQ. INDX1) GO TO 180
            INDXR1 = INDX1
            INDXR2 = INDX - 1
            INDXC1 = JU(JL(INDX))
            INDXC2 = JU(JL(INDX)+1) - 1
            IF(INDXC1 .GT. INDXC2) GO TO 180
 160        KR = JL(INDXR1)
 170        KC = IU(INDXC1)
            IF(KR .GT. KC) THEN
               INDXC1 = INDXC1 + 1
               IF(INDXC1 .LE. INDXC2) GO TO 170
            ELSEIF(KR .LT. KC) THEN
               INDXR1 = INDXR1 + 1
               IF(INDXR1 .LE. INDXR2) GO TO 160
            ELSEIF(KR .EQ. KC) THEN
               L(INDX) = L(INDX) - L(INDXR1)*DINV(KC)*U(INDXC1)
               INDXR1 = INDXR1 + 1
               INDXC1 = INDXC1 + 1
               IF(INDXR1 .LE. INDXR2 .AND. INDXC1 .LE. INDXC2) GO TO 160
            ENDIF
 180        L(INDX) = L(INDX)/DINV(JL(INDX))
 190     CONTINUE
C         
C         ith column of u
 200     INDX1 = JU(I)
         INDX2 = JU(I+1) - 1
         IF(INDX1 .GT. INDX2) GO TO 260
         DO 250 INDX=INDX1,INDX2
            IF(INDX .EQ. INDX1) GO TO 240
            INDXC1 = INDX1
            INDXC2 = INDX - 1
            INDXR1 = IL(IU(INDX))
            INDXR2 = IL(IU(INDX)+1) - 1
            IF(INDXR1 .GT. INDXR2) GO TO 240
 210        KR = JL(INDXR1)
 220        KC = IU(INDXC1)
            IF(KR .GT. KC) THEN
               INDXC1 = INDXC1 + 1
               IF(INDXC1 .LE. INDXC2) GO TO 220
            ELSEIF(KR .LT. KC) THEN
               INDXR1 = INDXR1 + 1
               IF(INDXR1 .LE. INDXR2) GO TO 210
            ELSEIF(KR .EQ. KC) THEN
               U(INDX) = U(INDX) - L(INDXR1)*DINV(KC)*U(INDXC1)
               INDXR1 = INDXR1 + 1
               INDXC1 = INDXC1 + 1
               IF(INDXR1 .LE. INDXR2 .AND. INDXC1 .LE. INDXC2) GO TO 210
            ENDIF
 240        U(INDX) = U(INDX)/DINV(IU(INDX))
 250     CONTINUE
C         
C         ith diagonal element
 260     INDXR1 = IL(I)
         INDXR2 = IL(I+1) - 1
         IF(INDXR1 .GT. INDXR2) GO TO 300
         INDXC1 = JU(I)
         INDXC2 = JU(I+1) - 1
         IF(INDXC1 .GT. INDXC2) GO TO 300
 270     KR = JL(INDXR1)
 280     KC = IU(INDXC1)
         IF(KR .GT. KC) THEN
            INDXC1 = INDXC1 + 1
            IF(INDXC1 .LE. INDXC2) GO TO 280
         ELSEIF(KR .LT. KC) THEN
            INDXR1 = INDXR1 + 1
            IF(INDXR1 .LE. INDXR2) GO TO 270
         ELSEIF(KR .EQ. KC) THEN
            DINV(I) = DINV(I) - L(INDXR1)*DINV(KC)*U(INDXC1)
            INDXR1 = INDXR1 + 1
            INDXC1 = INDXC1 + 1
            IF(INDXR1 .LE. INDXR2 .AND. INDXC1 .LE. INDXC2) GO TO 270
         ENDIF
C         
 300  CONTINUE
C         
C         replace diagonal lts by their inverses.
CVD$ VECTOR
      DO 430 I=1,N
         DINV(I) = 1./DINV(I)
 430  CONTINUE
C         
      RETURN
C------------- LAST LINE OF DSILUS FOLLOWS ----------------------------
      END
      SUBROUTINE DSLUI2(N, B, X, IL, JL, L, DINV, IU, JU, U )
C***BEGIN PROLOGUE  DSLUI2
C***DATE WRITTEN   871119   (YYMMDD)
C***REVISION DATE  881213   (YYMMDD)
C***CATEGORY NO.  D2A4
C***KEYWORDS  LIBRARY=SLATEC(SLAP),
C             TYPE=DOUBLE PRECISION(DSLUI2-S),
C             Non-Symmetric Linear system solve, Sparse, 
C             Iterative Precondition
C***AUTHOR  Greenbaum, Anne, Courant Institute
C           Seager, Mark K., (LLNL)
C             Lawrence Livermore National Laboratory
C             PO BOX 808, L-300
C             Livermore, CA 94550 (415) 423-3141
C             seager@lll-crg.llnl.gov
C***PURPOSE  SLAP Back solve for LDU Factorization.
C            Routine  to  solve a system of the form  L*D*U X  =  B,
C            where L is a unit  lower  triangular  matrix,  D  is  a 
C            diagonal matrix, and U is a unit upper triangular matrix.
C***DESCRIPTION
C *Usage:
C     INTEGER N, IL(N+1), JL(NL), IU(NU), JU(N+1)
C     DOUBLE PRECISION B(N), X(N), L(NL), DINV(N), U(NU)
C
C     CALL DSLUI2( N, B, X, IL, JL, L, DINV, IU, JU, U )
C
C *Arguments:
C N      :IN       Integer
C         Order of the Matrix.
C B      :IN       Double Precision B(N).
C         Right hand side.
C X      :OUT      Double Precision X(N).
C         Solution of L*D*U x = b.
C NEL    :IN       Integer.
C         Number of non-zeros in the EL array.
C IL     :IN       Integer IL(N+1).
C JL     :IN       Integer JL(NL).
C  L     :IN       Double Precision L(NL).
C         IL, JL, L contain the unit  lower triangular factor of the
C         incomplete decomposition of some matrix stored in SLAP Row
C         format.  The diagonal of ones *IS* stored.  This structure
C         can   be   set up  by   the  DSILUS routine.   See 
C         "DESCRIPTION", below  for more   details about   the  SLAP
C         format.
C DINV   :IN       Double Precision DINV(N).
C         Inverse of the diagonal matrix D.
C NU     :IN       Integer.
C         Number of non-zeros in the U array.     
C IU     :IN       Integer IU(N+1).
C JU     :IN       Integer JU(NU).
C U      :IN       Double Precision U(NU).
C         IU, JU, U contain the unit upper triangular factor  of the
C         incomplete decomposition  of  some  matrix stored in  SLAP
C         Column format.   The diagonal of ones  *IS* stored.   This
C         structure can be set up  by the DSILUS routine.  See
C         "DESCRIPTION", below   for  more   details about  the SLAP
C         format.
C
C *Description:
C       This routine is supplied with  the SLAP package as a routine
C       to  perform  the  MSOLVE operation  in   the  SIR and   SBCG
C       iteration routines for  the  drivers DSILUR and DSLUBC.   It
C       must  be called  via   the  SLAP  MSOLVE  calling   sequence
C       convention interface routine DSLUI.
C         **** THIS ROUTINE ITSELF DOES NOT CONFORM TO THE ****
C               **** SLAP MSOLVE CALLING CONVENTION ****
C
C       IL, JL, L should contain the unit lower triangular factor of
C       the incomplete decomposition of the A matrix  stored in SLAP
C       Row format.  IU, JU, U should contain  the unit upper factor
C       of the  incomplete decomposition of  the A matrix  stored in
C       SLAP Column format This ILU factorization can be computed by
C       the DSILUS routine.  The diagonals (which is all one's) are
C       stored.
C
C       =================== S L A P Column format ==================
C       This routine requires  that the  matrix  A be  stored in the
C       SLAP Column format.  In this format the non-zeros are stored
C       counting down columns (except for  the diagonal entry, which
C       must appear first in each  "column")  and are stored  in the
C       double precision array A.   In other words,  for each column
C       in the matrix put the diagonal entry in  A.  Then put in the
C       other non-zero  elements going down  the column (except  the
C       diagonal) in order.   The  IA array holds the  row index for
C       each non-zero.  The JA array holds the offsets  into the IA,
C       A arrays  for  the  beginning  of each   column.   That  is,
C       IA(JA(ICOL)),  A(JA(ICOL)) points   to the beginning  of the
C       ICOL-th   column    in    IA and   A.      IA(JA(ICOL+1)-1),
C       A(JA(ICOL+1)-1) points to  the  end of the   ICOL-th column.
C       Note that we always have  JA(N+1) = NELT+1,  where N is  the
C       number of columns in  the matrix and NELT  is the number  of
C       non-zeros in the matrix.
C       
C       Here is an example of the  SLAP Column  storage format for a
C       5x5 Matrix (in the A and IA arrays '|'  denotes the end of a 
C       column):
C       
C       5x5 Matrix      SLAP Column format for 5x5 matrix on left.
C       1  2  3    4  5    6  7    8    9 10 11
C       |11 12  0  0 15|   A: 11 21 51 | 22 12 | 33 53 | 44 | 55 15 35
C       |21 22  0  0  0|  IA:  1  2  5 |  2  1 |  3  5 |  4 |  5  1  3
C       | 0  0 33  0 35|  JA:  1  4  6    8  9   12
C       | 0  0  0 44  0|
C       |51  0 53  0 55|
C       
C       ==================== S L A P Row format ====================
C       This routine requires  that the matrix A  be  stored  in the
C       SLAP  Row format.   In this format  the non-zeros are stored
C       counting across  rows (except for the diagonal  entry, which
C       must appear first in each "row") and  are stored in the 
C       double precision
C       array A.  In other words, for each row in the matrix put the
C       diagonal entry in  A.   Then   put  in the   other  non-zero
C       elements   going  across the  row (except   the diagonal) in
C       order.   The  JA array  holds   the column   index for  each
C       non-zero.   The IA  array holds the  offsets into  the JA, A
C       arrays  for   the   beginning  of   each  row.   That    is,
C       JA(IA(IROW)),  A(IA(IROW)) points  to  the beginning  of the
C       IROW-th row in JA and A.   JA(IA(IROW+1)-1), A(IA(IROW+1)-1)
C       points to the  end of the  IROW-th row.  Note that we always
C       have IA(N+1) =  NELT+1, where  N  is  the number of rows  in
C       the matrix  and NELT  is the  number   of  non-zeros in  the
C       matrix.
C       
C       Here is an example of the SLAP Row storage format for a  5x5
C       Matrix (in the A and JA arrays '|' denotes the end of a row):
C
C           5x5 Matrix         SLAP Row format for 5x5 matrix on left.
C                              1  2  3    4  5    6  7    8    9 10 11
C       |11 12  0  0 15|   A: 11 12 15 | 22 21 | 33 35 | 44 | 55 51 53
C       |21 22  0  0  0|  JA:  1  2  5 |  2  1 |  3  5 |  4 |  5  1  3
C       | 0  0 33  0 35|  IA:  1  4  6    8  9   12
C       | 0  0  0 44  0|  
C       |51  0 53  0 55|  
C
C       With  the SLAP  format  the "inner  loops" of  this  routine
C       should vectorize   on machines with   hardware  support  for
C       vector gather/scatter operations.  Your compiler may require
C       a  compiler directive  to  convince   it that there  are  no
C       implicit vector  dependencies.  Compiler directives  for the
C       Alliant FX/Fortran and CRI CFT/CFT77 compilers  are supplied
C       with the standard SLAP distribution.
C
C *Precision:           Double Precision
C *See Also:
C       DSILUS
C***REFERENCES  (NONE)
C***ROUTINES CALLED  (NONE)
C***END PROLOGUE  DSLUI2
      IMPLICIT DOUBLE PRECISION(A-H,O-Z)
      INTEGER N, IL(1), JL(1), IU(1), JU(1)
      DOUBLE PRECISION B(N), X(N), L(1), DINV(N), U(1)
C         
C         Solve  L*Y = B,  storing result in X, L stored by rows.
C***FIRST EXECUTABLE STATEMENT  DSLUI2
      DO 10 I = 1, N
         X(I) = B(I)
 10   CONTINUE
      DO 30 IROW = 2, N
         JBGN = IL(IROW)
         JEND = IL(IROW+1)-1
         IF( JBGN.LE.JEND ) THEN
CLLL. OPTION ASSERT (NOHAZARD)
CDIR$ IVDEP
CVD$ ASSOC
CVD$ NODEPCHK
            DO 20 J = JBGN, JEND
               X(IROW) = X(IROW) - L(J)*X(JL(J))
 20         CONTINUE
         ENDIF
 30   CONTINUE
C         
C         Solve  D*Z = Y,  storing result in X.
      DO 40 I=1,N
         X(I) = X(I)*DINV(I)
 40   CONTINUE
C         
C         Solve  U*X = Z, U stored by columns.
      DO 60 ICOL = N, 2, -1
         JBGN = JU(ICOL)
         JEND = JU(ICOL+1)-1
         IF( JBGN.LE.JEND ) THEN
CLLL. OPTION ASSERT (NOHAZARD)
CDIR$ IVDEP
CVD$ NODEPCHK
            DO 50 J = JBGN, JEND
               X(IU(J)) = X(IU(J)) - U(J)*X(ICOL)
 50         CONTINUE
         ENDIF
 60   CONTINUE
C         
      RETURN
C------------- LAST LINE OF DSLUI2 FOLLOWS ----------------------------
      END
      SUBROUTINE DS2Y(N, NELT, IA, JA, A, ISYM )
C***BEGIN PROLOGUE  DS2Y
C***DATE WRITTEN   871119   (YYMMDD)
C***REVISION DATE  881213   (YYMMDD)
C***CATEGORY NO.  D2A4, D2B4
C***KEYWORDS  LIBRARY=SLATEC(SLAP),
C             TYPE=DOUBLE PRECISION(DS2Y-D),
C             Linear system, SLAP Sparse
C***AUTHOR  Seager, Mark K., (LLNL)
C             Lawrence Livermore National Laboratory
C             PO BOX 808, L-300
C             Livermore, CA 94550 (415) 423-3141
C             seager@lll-crg.llnl.gov
C***PURPOSE  SLAP Triad to SLAP Column Format Converter.
C            Routine to convert from the SLAP Triad to SLAP Column
C            format.
C***DESCRIPTION
C *Usage:
C     INTEGER N, NELT, IA(NELT), JA(NELT), ISYM
C     DOUBLE PRECISION A(NELT)
C
C     CALL DS2Y( N, NELT, IA, JA, A, ISYM )
C
C *Arguments:
C N      :IN       Integer
C         Order of the Matrix.
C NELT   :IN       Integer.
C         Number of non-zeros stored in A.
C IA     :INOUT    Integer IA(NELT).
C JA     :INOUT    Integer JA(NELT).
C A      :INOUT    Double Precision A(NELT).
C         These arrays should hold the matrix A in either the SLAP
C         Triad format or the SLAP Column format.  See "LONG 
C         DESCRIPTION", below.  If the SLAP Triad format is used
C         this format is translated to the SLAP Column format by
C         this routine.
C ISYM   :IN       Integer.
C         Flag to indicate symmetric storage format.
C         If ISYM=0, all nonzero entries of the matrix are stored.
C         If ISYM=1, the matrix is symmetric, and only the lower
C         triangle of the matrix is stored.
C
C *Precision:           Double Precision
C
C***LONG DESCRIPTION
C       The Sparse Linear Algebra Package (SLAP) utilizes two matrix
C       data structures: 1) the  SLAP Triad  format or  2)  the SLAP
C       Column format.  The user can hand this routine either of the
C       of these data structures.  If the SLAP Triad format is give
C       as input then this routine transforms it into SLAP Column
C       format.  The way this routine tells which format is given as
C       input is to look at JA(N+1).  If JA(N+1) = NELT+1 then we
C       have the SLAP Column format.  If that equality does not hold
C       then it is assumed that the IA, JA, A arrays contain the 
C       SLAP Triad format.
C       
C       =================== S L A P Triad format ===================
C       This routine requires that the  matrix A be   stored in  the
C       SLAP  Triad format.  In  this format only the non-zeros  are
C       stored.  They may appear in  *ANY* order.  The user supplies
C       three arrays of  length NELT, where  NELT is  the number  of
C       non-zeros in the matrix: (IA(NELT), JA(NELT), A(NELT)).  For
C       each non-zero the user puts the row and column index of that
C       matrix element  in the IA and  JA arrays.  The  value of the
C       non-zero   matrix  element is  placed  in  the corresponding
C       location of the A array.   This is  an  extremely  easy data
C       structure to generate.  On  the  other hand it   is  not too
C       efficient on vector computers for  the iterative solution of
C       linear systems.  Hence,   SLAP changes   this  input    data
C       structure to the SLAP Column format  for  the iteration (but
C       does not change it back).
C       
C       Here is an example of the  SLAP Triad   storage format for a
C       5x5 Matrix.  Recall that the entries may appear in any order.
C
C           5x5 Matrix       SLAP Triad format for 5x5 matrix on left.
C                              1  2  3  4  5  6  7  8  9 10 11
C       |11 12  0  0 15|   A: 51 12 11 33 15 53 55 22 35 44 21
C       |21 22  0  0  0|  IA:  5  1  1  3  1  5  5  2  3  4  2
C       | 0  0 33  0 35|  JA:  1  2  1  3  5  3  5  2  5  4  1
C       | 0  0  0 44  0|
C       |51  0 53  0 55|
C       
C       =================== S L A P Column format ==================
C       This routine requires  that the  matrix  A be  stored in the
C       SLAP Column format.  In this format the non-zeros are stored
C       counting down columns (except for  the diagonal entry, which
C       must appear first in each  "column")  and are stored  in the
C       double precision array A.   In other words,  for each column
C       in the matrix put the diagonal entry in  A.  Then put in the
C       other non-zero  elements going down  the column (except  the
C       diagonal) in order.   The  IA array holds the  row index for
C       each non-zero.  The JA array holds the offsets  into the IA,
C       A arrays  for  the  beginning  of each   column.   That  is,
C       IA(JA(ICOL)),  A(JA(ICOL)) points   to the beginning  of the
C       ICOL-th   column    in    IA and   A.      IA(JA(ICOL+1)-1),
C       A(JA(ICOL+1)-1) points to  the  end of the   ICOL-th column.
C       Note that we always have  JA(N+1) = NELT+1,  where N is  the
C       number of columns in  the matrix and NELT  is the number  of
C       non-zeros in the matrix.
C       
C       Here is an example of the  SLAP Column  storage format for a
C       5x5 Matrix (in the A and IA arrays '|'  denotes the end of a 
C       column):
C       
C       5x5 Matrix      SLAP Column format for 5x5 matrix on left.
C       1  2  3    4  5    6  7    8    9 10 11
C       |11 12  0  0 15|   A: 11 21 51 | 22 12 | 33 53 | 44 | 55 15 35
C       |21 22  0  0  0|  IA:  1  2  5 |  2  1 |  3  5 |  4 |  5  1  3
C       | 0  0 33  0 35|  JA:  1  4  6    8  9   12
C       | 0  0  0 44  0|
C       |51  0 53  0 55|
C       
C***REFERENCES  (NONE)
C***ROUTINES CALLED  QS2I1D
C***END PROLOGUE  DS2Y
      IMPLICIT DOUBLE PRECISION(A-H,O-Z)
      INTEGER N, NELT, IA(NELT), JA(NELT), ISYM
      DOUBLE PRECISION A(NELT)
C
C         Check to see if the (IA,JA,A) arrays are in SLAP Column 
C         format.  If it's not then transform from SLAP Triad.
C***FIRST EXECUTABLE STATEMENT  DS2LT
      IF( JA(N+1).EQ.NELT+1 ) RETURN
C
C         Sort into ascending order by COLUMN (on the ja array).
C         This will line up the columns.
C
      CALL QS2I1D( JA, IA, A, NELT, 1 )
C         
C         Loop over each column to see where the column indicies change 
C         in the column index array ja.  This marks the beginning of the
C         next column.
C         
CVD$R NOVECTOR
      JA(1) = 1
      DO 20 ICOL = 1, N-1
         DO 10 J = JA(ICOL)+1, NELT
            IF( JA(J).NE.ICOL ) THEN
               JA(ICOL+1) = J
               GOTO 20
            ENDIF
 10      CONTINUE
 20   CONTINUE
      JA(N+1) = NELT+1
C         
C         Mark the n+2 element so that future calls to a SLAP routine 
C         utilizing the YSMP-Column storage format will be able to tell.
C         
      JA(N+2) = 0
C
C         Now loop thru the ia(i) array making sure that the Diagonal
C         matrix element appears first in the column.  Then sort the
C         rest of the column in ascending order.
C
      DO 70 ICOL = 1, N
         IBGN = JA(ICOL)
         IEND = JA(ICOL+1)-1
         DO 30 I = IBGN, IEND
            IF( IA(I).EQ.ICOL ) THEN
C         Swap the diag element with the first element in the column.
               ITEMP = IA(I)
               IA(I) = IA(IBGN)
               IA(IBGN) = ITEMP
               TEMP = A(I)
               A(I) = A(IBGN)
               A(IBGN) = TEMP
               GOTO 40
            ENDIF
 30      CONTINUE
 40      IBGN = IBGN + 1
         IF( IBGN.LT.IEND ) THEN
            DO 60 I = IBGN, IEND
               DO 50 J = I+1, IEND
                  IF( IA(I).GT.IA(J) ) THEN
                     ITEMP = IA(I)
                     IA(I) = IA(J)
                     IA(J) = ITEMP
                     TEMP = A(I)
                     A(I) = A(J)
                     A(J) = TEMP
                  ENDIF
 50            CONTINUE
 60         CONTINUE
         ENDIF
 70   CONTINUE
      RETURN
C------------- LAST LINE OF DS2Y FOLLOWS ----------------------------
      END
      SUBROUTINE QS2I1D( IA, JA, A, N, KFLAG )
C***BEGIN PROLOGUE  QS2I1D
C***DATE WRITTEN   761118   (YYMMDD)
C***REVISION DATE  890125   (YYMMDD)
C***CATEGORY NO.  N6A2A
C***KEYWORDS  LIBRARY=SLATEC(SLAP),
C             TYPE=INTEGER(QS2I1D-I),
C             QUICKSORT,DOUBLETON QUICKSORT,SORT,SORTING
C***AUTHOR  Jones, R. E., (SNLA)
C           Kahaner, D. K., (NBS)
C           Seager, M. K., (LLNL) seager@lll-crg.llnl.gov
C           Wisniewski, J. A., (SNLA)
C***PURPOSE  Sort an integer array also moving an integer and DP array
C            This routine sorts the integer  array  IA and makes the
C            same interchanges   in the integer   array  JA  and the
C            double precision array A.  The  array IA may be  sorted
C            in increasing order or decreas- ing  order.  A slightly
C            modified QUICKSORT algorithm is used.
C
C***DESCRIPTION
C     Written by Rondall E Jones
C     Modified by John A. Wisniewski to use the Singleton QUICKSORT
C     algorithm. date 18 November 1976.
C
C     Further modified by David K. Kahaner
C     National Bureau of Standards
C     August, 1981
C
C     Even further modification made to bring the code up to the 
C     Fortran 77 level and make it more readable and to carry
C     along one integer array and one double precision array during 
C     the sort by
C     Mark K. Seager
C     Lawrence Livermore National Laboratory
C     November, 1987
C     This routine was adapted from the ISORT routine.
C
C     ABSTRACT
C         This routine sorts an integer array IA and makes the same
C         interchanges in the integer array JA and the double precision
C          array A.  
C         The array a may be sorted in increasing order or decreasing 
C         order.  A slightly modified quicksort algorithm is used.
C
C     DESCRIPTION OF PARAMETERS
C        IA - Integer array of values to be sorted.
C        JA - Integer array to be carried along.
C         A - Double Precision array to be carried along.
C         N - Number of values in integer array IA to be sorted.
C     KFLAG - Control parameter
C           = 1 means sort IA in INCREASING order.
C           =-1 means sort IA in DECREASING order.
C
C***REFERENCES
C     Singleton, R. C., Algorithm 347, "An Efficient Algorithm for 
C     Sorting with Minimal Storage", cacm, Vol. 12, No. 3, 1969, 
C     Pp. 185-187.
C***ROUTINES CALLED  XERROR
C***END PROLOGUE  QS2I1D
      IMPLICIT DOUBLE PRECISION(A-H,O-Z)
CVD$R NOVECTOR
CVD$R NOCONCUR
      DIMENSION IL(21),IU(21)
      INTEGER   IA(N),JA(N),IT,IIT,JT,JJT
      DOUBLE PRECISION A(N), TA, TTA
C
C***FIRST EXECUTABLE STATEMENT  QS2I1D
      NN = N
      IF (NN.LT.1) THEN
         CALL XERROR ( 'QS2I1D- the number of values to be sorted was no
     $T POSITIVE.',59,1,1)
         RETURN
      ENDIF
      IF( N.EQ.1 ) RETURN
      KK = IABS(KFLAG)
      IF ( KK.NE.1 ) THEN
         CALL XERROR ( 'QS2I1D- the sort control parameter, k, was not 1
     $ OR -1.',55,2,1)
         RETURN
      ENDIF
C
C     Alter array IA to get decreasing order if needed.
C
      IF( KFLAG.LT.1 ) THEN
         DO 20 I=1,NN
            IA(I) = -IA(I)
 20      CONTINUE
      ENDIF
C
C     Sort IA and carry JA and A along.
C     And now...Just a little black magic...
      M = 1
      I = 1
      J = NN
      R = .375
 210  IF( R.LE.0.5898437 ) THEN
         R = R + 3.90625E-2
      ELSE
         R = R-.21875
      ENDIF
 225  K = I
C
C     Select a central element of the array and save it in location 
C     it, jt, at.
C
      IJ = I + IDINT( DBLE(J-I)*R )
      IT = IA(IJ)
      JT = JA(IJ)
      TA = A(IJ)
C
C     If first element of array is greater than it, interchange with it.
C
      IF( IA(I).GT.IT ) THEN
         IA(IJ) = IA(I)
         IA(I)  = IT
         IT     = IA(IJ)
         JA(IJ) = JA(I)
         JA(I)  = JT
         JT     = JA(IJ)
         A(IJ)  = A(I)
         A(I)   = TA
         TA     = A(IJ)
      ENDIF
      L=J
C                           
C     If last element of array is less than it, swap with it.
C
      IF( IA(J).LT.IT ) THEN
         IA(IJ) = IA(J)
         IA(J)  = IT
         IT     = IA(IJ)
         JA(IJ) = JA(J)
         JA(J)  = JT
         JT     = JA(IJ)
         A(IJ)  = A(J)
         A(J)   = TA
         TA     = A(IJ)
C
C     If first element of array is greater than it, swap with it.
C
         IF ( IA(I).GT.IT ) THEN
            IA(IJ) = IA(I)
            IA(I)  = IT
            IT     = IA(IJ)
            JA(IJ) = JA(I)
            JA(I)  = JT
            JT     = JA(IJ)
            A(IJ)  = A(I)
            A(I)   = TA
            TA     = A(IJ)
         ENDIF
      ENDIF
C
C     Find an element in the second half of the array which is 
C     smaller than it.
C
  240 L=L-1
      IF( IA(L).GT.IT ) GO TO 240
C
C     Find an element in the first half of the array which is 
C     greater than it.
C
  245 K=K+1
      IF( IA(K).LT.IT ) GO TO 245
C
C     Interchange these elements.
C
      IF( K.LE.L ) THEN
         IIT   = IA(L)
         IA(L) = IA(K)
         IA(K) = IIT
         JJT   = JA(L)
         JA(L) = JA(K)
         JA(K) = JJT
         TTA   = A(L)
         A(L)  = A(K)
         A(K)  = TTA
         GOTO 240
      ENDIF
C
C     Save upper and lower subscripts of the array yet to be sorted.
C
      IF( L-I.GT.J-K ) THEN
         IL(M) = I
         IU(M) = L
         I = K
         M = M+1
      ELSE
         IL(M) = K
         IU(M) = J
         J = L
         M = M+1
      ENDIF
      GO TO 260
C
C     Begin again on another portion of the unsorted array.
C                                  
  255 M = M-1
      IF( M.EQ.0 ) GO TO 300
      I = IL(M)
      J = IU(M)
  260 IF( J-I.GE.1 ) GO TO 225
      IF( I.EQ.J ) GO TO 255
      IF( I.EQ.1 ) GO TO 210
      I = I-1
  265 I = I+1
      IF( I.EQ.J ) GO TO 255
      IT = IA(I+1)
      JT = JA(I+1)
      TA =  A(I+1)
      IF( IA(I).LE.IT ) GO TO 265
      K=I
  270 IA(K+1) = IA(K)
      JA(K+1) = JA(K)
      A(K+1)  =  A(K)
      K = K-1
      IF( IT.LT.IA(K) ) GO TO 270
      IA(K+1) = IT
      JA(K+1) = JT
      A(K+1)  = TA
      GO TO 265
C
C     Clean up, if necessary.
C
  300 IF( KFLAG.LT.1 ) THEN
         DO 310 I=1,NN
            IA(I) = -IA(I)
 310     CONTINUE
      ENDIF
      RETURN
C------------- LAST LINE OF QS2I1D FOLLOWS ----------------------------
      END
      SUBROUTINE XERABT(MESSG,NMESSG)
C***begin prologue  xerabt
C***date written   790801   (yymmdd)
C***revision date  851111   (yymmdd)
C***category no.  r3c
C***keywords  error,xerror package
C***author  jones, r. e., (snla)
C***purpose  abort program execution and print error message.
C***description
C
C     abstract
C        ***note*** machine dependent routine
C        xerabt aborts the execution of the program.
C        the error message causing the abort is given in the calling
C        sequence, in case one needs it for printing on a dayfile,
C        for example.
C
C     description of parameters
C        messg and nmessg are as in xerror, except that nmessg may
C        be zero, in which case no message is being supplied.
C
C     written by ron jones, with slatec common math library subcommittee
C     latest revision ---  1 august 1982
C***references  jones r.e., kahaner d.k., 'xerror, the slatec error-
C                 handling package', sand82-0800, sandia laboratories,
C                 1982.
C***routines called  (none)
C***end prologue  xerabt
      CHARACTER*(*) MESSG
C***first executable statement  xerabt
      CALL EXIT(1)
      END
      SUBROUTINE XERCTL(MESSG1,NMESSG,NERR,LEVEL,KONTRL)
C***begin prologue  xerctl
C***date written   790801   (yymmdd)
C***revision date  851111   (yymmdd)
C***category no.  r3c
C***keywords  error,xerror package
C***author  jones, r. e., (snla)
C***purpose  allow user control over handling of errors.
C***description
C
C     abstract
C        allows user control over handling of individual errors.
C        just after each message is recorded, but before it is
C        processed any further (i.e., before it is printed or
C        a decision to abort is made), a call is made to xerctl.
C        if the user has provided his own version of xerctl, he
C        can then override the value of kontrol used in processing
C        this message by redefining its value.
C        kontrl may be set to any value from -2 to 2.
C        the meanings for kontrl are the same as in xsetf, except
C        that the value of kontrl changes only for this message.
C        if kontrl is set to a value outside the range from -2 to 2,
C        it will be moved back into that range.
C
C     description of parameters
C
C      --input--
C        messg1 - the first word (only) of the error message.
C        nmessg - same as in the call to xerror or xerrwv.
C        nerr   - same as in the call to xerror or xerrwv.
C        level  - same as in the call to xerror or xerrwv.
C        kontrl - the current value of the control flag as set
C                 by a call to xsetf.
C
C      --output--
C        kontrl - the new value of kontrl.  if kontrl is not
C                 defined, it will remain at its original value.
C                 this changed value of control affects only
C                 the current occurrence of the current message.
C***references  jones r.e., kahaner d.k., 'xerror, the slatec error-
C                 handling package', sand82-0800, sandia laboratories,
C                 1982.
C***routines called  (none)
C***end prologue  xerctl
      CHARACTER*20 MESSG1
C***first executable statement  xerctl
      RETURN
      END
      SUBROUTINE XERPRT(MESSG,NMESSG)
C***begin prologue  xerprt
C***date written   790801   (yymmdd)
C***revision date  851213   (yymmdd)
C***category no.  r3
C***keywords  error,xerror package
C***author  jones, r. e., (snla)
C***purpose  print error messages.
C***description
C
C     abstract
C        print the hollerith message in messg, of length nmessg,
C        on each file indicated by xgetua.
C     latest revision ---  1 august 1985
C***references  jones r.e., kahaner d.k., 'xerror, the slatec error-
C                 handling package', sand82-0800, sandia laboratories,
C                 1982.
C***routines called  i1mach,xgetua
C***end prologue  xerprt
      INTEGER LUN(5)
      CHARACTER*(*) MESSG
C     obtain unit numbers and write line to each unit
C***first executable statement  xerprt
      CALL XGETUA(LUN,NUNIT)
      LENMES = LEN(MESSG)
      DO 20 KUNIT=1,NUNIT
         IUNIT = LUN(KUNIT)
         IF (IUNIT.EQ.0) IUNIT = I1MACH(4)
         DO 10 ICHAR=1,LENMES,72
            LAST = MIN0(ICHAR+71 , LENMES)
            WRITE (IUNIT,'(1X,A)') MESSG(ICHAR:LAST)
   10    CONTINUE
   20 CONTINUE
      RETURN
      END
      SUBROUTINE XERROR(MESSG,NMESSG,NERR,LEVEL)
C***begin prologue  xerror
C***date written   790801   (yymmdd)
C***revision date  851111   (yymmdd)
C***category no.  r3c
C***keywords  error,xerror package
C***author  jones, r. e., (snla)
C***purpose  process an error (diagnostic) message.
C***description
C
C     abstract
C        xerror processes a diagnostic message, in a manner
C        determined by the value of level and the current value
C        of the library error control flag, kontrl.
C        (see subroutine xsetf for details.)
C
C     description of parameters
C      --input--
C        messg - the hollerith message to be processed, containing
C                no more than 72 characters.
C        nmessg- the actual number of characters in messg.
C        nerr  - the error number associated with this message.
C                nerr must not be zero.
C        level - error category.
C                =2 means this is an unconditionally fatal error.
C                =1 means this is a recoverable error.  (i.e., it is
C                   non-fatal if xsetf has been appropriately called.)
C                =0 means this is a warning message only.
C                =-1 means this is a warning message which is to be
C                   printed at most once, regardless of how many
C                   times this call is executed.
C
C     examples
C        call xerror('smooth -- num was zero.',23,1,2)
C        call xerror('integ  -- less than full accuracy achieved.',
C    1                43,2,1)
C        call xerror('rooter -- actual zero of f found before interval f
C    1ully collapsed.',65,3,0)
C        call xerror('exp    -- underflows being set to zero.',39,1,-1)
C
C     written by ron jones, with slatec common math library subcommittee
C***references  jones r.e., kahaner d.k., 'xerror, the slatec error-
C                 handling package', sand82-0800, sandia laboratories,
C                 1982.
C***routines called  xerrwv
C***end prologue  xerror
      CHARACTER*(*) MESSG
C***first executable statement  xerror
      CALL XERRWV(MESSG,NMESSG,NERR,LEVEL,0,0,0,0,0.,0.)
      RETURN
      END
      SUBROUTINE XERRWV(MESSG,NMESSG,NERR,LEVEL,NI,I1,I2,NR,R1,R2)
C***begin prologue  xerrwv
C***date written   800319   (yymmdd)
C***revision date  851111   (yymmdd)
C***category no.  r3c
C***keywords  error,xerror package
C***author  jones, r. e., (snla)
C***purpose  process an error message allowing 2 integer and 2 real
C            values to be included in the message.
C***description
C
C     abstract
C        xerrwv processes a diagnostic message, in a manner
C        determined by the value of level and the current value
C        of the library error control flag, kontrl.
C        (see subroutine xsetf for details.)
C        in addition, up to two integer values and two real
C        values may be printed along with the message.
C
C     description of parameters
C      --input--
C        messg - the hollerith message to be processed.
C        nmessg- the actual number of characters in messg.
C        nerr  - the error number associated with this message.
C                nerr must not be zero.
C        level - error category.
C                =2 means this is an unconditionally fatal error.
C                =1 means this is a recoverable error.  (i.e., it is
C                   non-fatal if xsetf has been appropriately called.)
C                =0 means this is a warning message only.
C                =-1 means this is a warning message which is to be
C                   printed at most once, regardless of how many
C                   times this call is executed.
C        ni    - number of integer values to be printed. (0 to 2)
C        i1    - first integer value.
C        i2    - second integer value.
C        nr    - number of real values to be printed. (0 to 2)
C        r1    - first real value.
C        r2    - second real value.
C
C     examples
C        call xerrwv('smooth -- num (=i1) was zero.',29,1,2,
C    1   1,num,0,0,0.,0.)
C        call xerrwv('quadxy -- requested error (r1) less than minimum (
C    1r2).,54,77,1,0,0,0,2,errreq,errmin)
C
C     latest revision ---  1 august 1985
C     written by ron jones, with slatec common math library subcommittee
C***references  jones r.e., kahaner d.k., 'xerror, the slatec error-
C                 handling package', sand82-0800, sandia laboratories,
C                 1982.
C***routines called  fdump,i1mach,j4save,xerabt,xerctl,xerprt,xersav,
C                    xgetua
C***end prologue  xerrwv
      CHARACTER*(*) MESSG
      CHARACTER*20 LFIRST
      CHARACTER*37 FORM
      DIMENSION LUN(5)
C     get flags
C***first executable statement  xerrwv
      LKNTRL = J4SAVE(2,0,.FALSE.)
      MAXMES = J4SAVE(4,0,.FALSE.)
C     check for valid input
      IF ((NMESSG.GT.0).AND.(NERR.NE.0).AND.
     1    (LEVEL.GE.(-1)).AND.(LEVEL.LE.2)) GO TO 10
         IF (LKNTRL.GT.0) CALL XERPRT('fatal error in...',17)
         CALL XERPRT('xerror -- invalid input',23)
C        IF (LKNTRL.GT.0) CALL FDUMP
         IF (LKNTRL.GT.0) CALL XERPRT('job abort due to fatal error.',
     1  29)
         IF (LKNTRL.GT.0) CALL XERSAV(' ',0,0,0,KDUMMY)
         CALL XERABT('xerror -- invalid input',23)
         RETURN
   10 CONTINUE
C     record message
      JUNK = J4SAVE(1,NERR,.TRUE.)
      CALL XERSAV(MESSG,NMESSG,NERR,LEVEL,KOUNT)
C     let user override
      LFIRST = MESSG
      LMESSG = NMESSG
      LERR = NERR
      LLEVEL = LEVEL
      CALL XERCTL(LFIRST,LMESSG,LERR,LLEVEL,LKNTRL)
C     reset to original values
      LMESSG = NMESSG
      LERR = NERR
      LLEVEL = LEVEL
      LKNTRL = MAX0(-2,MIN0(2,LKNTRL))
      MKNTRL = IABS(LKNTRL)
C     decide whether to print message
      IF ((LLEVEL.LT.2).AND.(LKNTRL.EQ.0)) GO TO 100
      IF (((LLEVEL.EQ.(-1)).AND.(KOUNT.GT.MIN0(1,MAXMES)))
     1.OR.((LLEVEL.EQ.0)   .AND.(KOUNT.GT.MAXMES))
     2.OR.((LLEVEL.EQ.1)   .AND.(KOUNT.GT.MAXMES).AND.(MKNTRL.EQ.1))
     3.OR.((LLEVEL.EQ.2)   .AND.(KOUNT.GT.MAX0(1,MAXMES)))) GO TO 100
         IF (LKNTRL.LE.0) GO TO 20
            CALL XERPRT(' ',1)
C           introduction
            IF (LLEVEL.EQ.(-1)) CALL XERPRT
     1('warning message...this message will only be printed once.',57)
            IF (LLEVEL.EQ.0) CALL XERPRT('warning in...',13)
            IF (LLEVEL.EQ.1) CALL XERPRT
     1      ('recoverable error in...',23)
            IF (LLEVEL.EQ.2) CALL XERPRT('fatal error in...',17)
   20    CONTINUE
C        message
         CALL XERPRT(MESSG,LMESSG)
         CALL XGETUA(LUN,NUNIT)
         ISIZEI = LOG10(FLOAT(I1MACH(9))) + 1.0
         ISIZEF = LOG10(FLOAT(I1MACH(10))**I1MACH(11)) + 1.0
         DO 50 KUNIT=1,NUNIT
            IUNIT = LUN(KUNIT)
            IF (IUNIT.EQ.0) IUNIT = I1MACH(4)
            DO 22 I=1,MIN(NI,2)
               WRITE (FORM,21) I,ISIZEI
   21          FORMAT ('(11x,21hin above message, i',I1,'=,i',I2,')   ')
               IF (I.EQ.1) WRITE (IUNIT,FORM) I1
               IF (I.EQ.2) WRITE (IUNIT,FORM) I2
   22       CONTINUE
            DO 24 I=1,MIN(NR,2)
               WRITE (FORM,23) I,ISIZEF+10,ISIZEF
   23          FORMAT ('(11x,21hin above message, r',I1,'=,e',
     1         I2,'.',I2,')')
               IF (I.EQ.1) WRITE (IUNIT,FORM) R1
               IF (I.EQ.2) WRITE (IUNIT,FORM) R2
   24       CONTINUE
            IF (LKNTRL.LE.0) GO TO 40
C              error number
               WRITE (IUNIT,30) LERR
   30          FORMAT (15H ERROR NUMBER =,I10)
   40       CONTINUE
   50    CONTINUE
C        trace-back
C        IF (LKNTRL.GT.0) CALL FDUMP
  100 CONTINUE
      IFATAL = 0
      IF ((LLEVEL.EQ.2).OR.((LLEVEL.EQ.1).AND.(MKNTRL.EQ.2)))
     1IFATAL = 1
C     quit here if message is not fatal
      IF (IFATAL.LE.0) RETURN
      IF ((LKNTRL.LE.0).OR.(KOUNT.GT.MAX0(1,MAXMES))) GO TO 120
C        print reason for abort
         IF (LLEVEL.EQ.1) CALL XERPRT
     1   ('job abort due to unrecovered error.',35)
         IF (LLEVEL.EQ.2) CALL XERPRT
     1   ('job abort due to fatal error.',29)
C        print error summary
         CALL XERSAV(' ',-1,0,0,KDUMMY)
  120 CONTINUE
C     abort
      IF ((LLEVEL.EQ.2).AND.(KOUNT.GT.MAX0(1,MAXMES))) LMESSG = 0
      CALL XERABT(MESSG,LMESSG)
      RETURN
      END
      SUBROUTINE XERSAV(MESSG,NMESSG,NERR,LEVEL,ICOUNT)
C***begin prologue  xersav
C***date written   800319   (yymmdd)
C***revision date  851213   (yymmdd)
C***category no.  r3
C***keywords  error,xerror package
C***author  jones, r. e., (snla)
C***purpose  record that an error has occurred.
C***description
C
C     abstract
C        record that this error occurred.
C
C     description of parameters
C     --input--
C       messg, nmessg, nerr, level are as in xerror,
C       except that when nmessg=0 the tables will be
C       dumped and cleared, and when nmessg is less than zero the
C       tables will be dumped and not cleared.
C     --output--
C       icount will be the number of times this message has
C       been seen, or zero if the table has overflowed and
C       does not contain this message specifically.
C       when nmessg=0, icount will not be altered.
C
C     written by ron jones, with slatec common math library subcommittee
C     latest revision ---  1 august 1985
C***references  jones r.e., kahaner d.k., 'xerror, the slatec error-
C                 handling package', sand82-0800, sandia laboratories,
C                 1982.
C***routines called  i1mach,xgetua
C***end prologue  xersav
      INTEGER LUN(5)
      CHARACTER*(*) MESSG
      CHARACTER*20 MESTAB(10),MES
      DIMENSION NERTAB(10),LEVTAB(10),KOUNT(10)
      SAVE MESTAB,NERTAB,LEVTAB,KOUNT,KOUNTX
C     next two data statements are necessary to provide a blank
C     error table initially
      DATA KOUNT(1),KOUNT(2),KOUNT(3),KOUNT(4),KOUNT(5),
     1     KOUNT(6),KOUNT(7),KOUNT(8),KOUNT(9),KOUNT(10)
     2     /0,0,0,0,0,0,0,0,0,0/
      DATA KOUNTX/0/
C***first executable statement  xersav
      IF (NMESSG.GT.0) GO TO 80
C     dump the table
         IF (KOUNT(1).EQ.0) RETURN
C        print to each unit
         CALL XGETUA(LUN,NUNIT)
         DO 60 KUNIT=1,NUNIT
            IUNIT = LUN(KUNIT)
            IF (IUNIT.EQ.0) IUNIT = I1MACH(4)
C           print table header
            WRITE (IUNIT,10)
   10       FORMAT (32H0          error message summary/
     1      51H message start             nerr     level     count)
C           print body of table
            DO 20 I=1,10
               IF (KOUNT(I).EQ.0) GO TO 30
               WRITE (IUNIT,15) MESTAB(I),NERTAB(I),LEVTAB(I),KOUNT(I)
   15          FORMAT (1X,A20,3I10)
   20       CONTINUE
   30       CONTINUE
C           print number of other errors
            IF (KOUNTX.NE.0) WRITE (IUNIT,40) KOUNTX
   40       FORMAT (41H0other errors not individually tabulated=,I10)
            WRITE (IUNIT,50)
   50       FORMAT (1X)
   60    CONTINUE
         IF (NMESSG.LT.0) RETURN
C        clear the error tables
         DO 70 I=1,10
   70       KOUNT(I) = 0
         KOUNTX = 0
         RETURN
   80 CONTINUE
C     process a message...
C     search for this messg, or else an empty slot for this messg,
C     or else determine that the error table is full.
      MES = MESSG
      DO 90 I=1,10
         II = I
         IF (KOUNT(I).EQ.0) GO TO 110
         IF (MES.NE.MESTAB(I)) GO TO 90
         IF (NERR.NE.NERTAB(I)) GO TO 90
         IF (LEVEL.NE.LEVTAB(I)) GO TO 90
         GO TO 100
   90 CONTINUE
C     three possible cases...
C     table is full
         KOUNTX = KOUNTX+1
         ICOUNT = 1
         RETURN
C     message found in table
  100    KOUNT(II) = KOUNT(II) + 1
         ICOUNT = KOUNT(II)
         RETURN
C     empty slot found for new message
  110    MESTAB(II) = MES
         NERTAB(II) = NERR
         LEVTAB(II) = LEVEL
         KOUNT(II)  = 1
         ICOUNT = 1
         RETURN
      END
      SUBROUTINE XGETUA(IUNITA,N)
C***begin prologue  xgetua
C***date written   790801   (yymmdd)
C***revision date  851111   (yymmdd)
C***category no.  r3c
C***keywords  error,xerror package
C***author  jones, r. e., (snla)
C***purpose  return unit number(s) to which error messages are being
C            sent.
C***description
C
C     abstract
C        xgetua may be called to determine the unit number or numbers
C        to which error messages are being sent.
C        these unit numbers may have been set by a call to xsetun,
C        or a call to xsetua, or may be a default value.
C
C     description of parameters
C      --output--
C        iunit - an array of one to five unit numbers, depending
C                on the value of n.  a value of zero refers to the
C                default unit, as defined by the i1mach machine
C                constant routine.  only iunit(1),...,iunit(n) are
C                defined by xgetua.  the values of iunit(n+1),...,
C                iunit(5) are not defined (for n .lt. 5) or altered
C                in any way by xgetua.
C        n     - the number of units to which copies of the
C                error messages are being sent.  n will be in the
C                range from 1 to 5.
C
C     latest revision ---  19 mar 1980
C     written by ron jones, with slatec common math library subcommittee
C***references  jones r.e., kahaner d.k., 'xerror, the slatec error-
C                 handling package', sand82-0800, sandia laboratories,
C                 1982.
C***routines called  j4save
C***end prologue  xgetua
      DIMENSION IUNITA(5)
C***first executable statement  xgetua
      N = J4SAVE(5,0,.FALSE.)
      DO 30 I=1,N
         INDEX = I+4
         IF (I.EQ.1) INDEX = 3
         IUNITA(I) = J4SAVE(INDEX,0,.FALSE.)
   30 CONTINUE
      RETURN
      END
      FUNCTION J4SAVE(IWHICH,IVALUE,ISET)
C***begin prologue  j4save
C***refer to  xerror
C***routines called  (none)
C***description
C
C     abstract
C        j4save saves and recalls several global variables needed
C        by the library error handling routines.
C
C     description of parameters
C      --input--
C        iwhich - index of item desired.
C                = 1 refers to current error number.
C                = 2 refers to current error control flag.
C                 = 3 refers to current unit number to which error
C                    messages are to be sent.  (0 means use standard.)
C                 = 4 refers to the maximum number of times any
C                     message is to be printed (as set by xermax).
C                 = 5 refers to the total number of units to which
C                     each error message is to be written.
C                 = 6 refers to the 2nd unit for error messages
C                 = 7 refers to the 3rd unit for error messages
C                 = 8 refers to the 4th unit for error messages
C                 = 9 refers to the 5th unit for error messages
C        ivalue - the value to be set for the iwhich-th parameter,
C                 if iset is .true. .
C        iset   - if iset=.true., the iwhich-th parameter will be
C                 given the value, ivalue.  if iset=.false., the
C                 iwhich-th parameter will be unchanged, and ivalue
C                 is a dummy parameter.
C      --output--
C        the (old) value of the iwhich-th parameter will be returned
C        in the function value, j4save.
C
C     written by ron jones, with slatec common math library subcommittee
C    adapted from bell laboratories port library error handler
C     latest revision ---  1 august 1985
C***references  jones r.e., kahaner d.k., 'xerror, the slatec error-
C                 handling package', sand82-0800, sandia laboratories,
C                 1982.
C***end prologue  j4save
      LOGICAL ISET
      INTEGER IPARAM(9)
      SAVE IPARAM
      DATA IPARAM(1),IPARAM(2),IPARAM(3),IPARAM(4)/0,2,0,10/
      DATA IPARAM(5)/1/
      DATA IPARAM(6),IPARAM(7),IPARAM(8),IPARAM(9)/0,0,0,0/
C***first executable statement  j4save
      J4SAVE = IPARAM(IWHICH)
      IF (ISET) IPARAM(IWHICH) = IVALUE
      RETURN
      END
      SUBROUTINE DSBDSC(N, NELT, IA, JA, A, ISYM, M, AINV, IPVT, IERR)
      INTEGER N, NELT
      INTEGER IA(NELT), JA(NELT)
      DOUBLE PRECISION A(NELT)
      INTEGER ISYM, M
      DOUBLE PRECISION AINV(M,N)
      INTEGER IPVT(M,N)
C
C     setup routine for block diagonal preconditioning.
C     uses LINPACK subroutine DGEFA for LU-decomposition of the
C     block matrices
C
      INTEGER NB, JBST, JBEND, JSA, JSE, IH
      DOUBLE PRECISION ZERO
      PARAMETER (ZERO=0.0D0)
      EXTERNAL DGEFA
      IF (MOD(N,M).NE.0) THEN
        IERR=-1
        RETURN
      ENDIF
      IERR=0
      DO 1 J=1,N
        DO 10 I=1,M
          AINV(I,J)=ZERO
          IPVT(I,J)=0
10      CONTINUE
1     CONTINUE
      NB=INT(N/M)
      DO 2 JB=1,NB
        JBST  = (JB-1)*M+1
        JBEND = JB*M
        DO 20 J=JBST,JBEND
          JSA = JA(J)
          JSE = JA(J+1)-1
          DO 200 IH=JSA,JSE
            I=IA(IH)
            IF (I.GE.JBST .AND. I.LE.JBEND) THEN
              AINV(I-JBST+1,J)=A(IH)
            ENDIF
200       CONTINUE
20      CONTINUE
        CALL DGEFA(AINV(1,JBST),M,M,IPVT(1,JBST),IERR)
        IF (IERR.NE.0) THEN
          IERR=IERR+JBST-1
          RETURN
        ENDIF
2     CONTINUE
      RETURN
      END
C
      SUBROUTINE DSBDIC(N, B, X, NELT, IA, JA, A, ISYM, M, AINV, IPVT)
      INTEGER N
      DOUBLE PRECISION B(N), X(N)
      INTEGER NELT
      INTEGER IA(NELT), JA(NELT)
      DOUBLE PRECISION A(NELT)
      INTEGER ISYM, M
      DOUBLE PRECISION AINV(M,N)
      INTEGER IPVT(M,N)
C
C     block diagonal preconditioner routine.
C     uses LINPACK subroutine DGESL for the solution of the linear
C     block systems
C
      EXTERNAL DGESL
      IF (MOD(N,M).NE.0) THEN
        IERR=-1
        RETURN
      ENDIF
      IERR=0
      DO 1 I=1,N
        X(I)=B(I)
1     CONTINUE
      DO 2 J=1,N,M
        CALL DGESL(AINV(1,J),M,M,IPVT(1,J),X(J),0)
2     CONTINUE
      RETURN
      END
C
C
      INTEGER FUNCTION I1MACH(I)
C
C  I/O UNIT NUMBERS.
C
C    I1MACH( 1) = THE STANDARD INPUT UNIT.
C
C    I1MACH( 2) = THE STANDARD OUTPUT UNIT.
C
C    I1MACH( 3) = THE STANDARD PUNCH UNIT.
C
C    I1MACH( 4) = THE STANDARD ERROR MESSAGE UNIT.
C
C  WORDS.
C
C    I1MACH( 5) = THE NUMBER OF BITS PER INTEGER STORAGE UNIT.
C
C    I1MACH( 6) = THE NUMBER OF CHARACTERS PER INTEGER STORAGE UNIT.
C
C  INTEGERS.
C
C    ASSUME INTEGERS ARE REPRESENTED IN THE S-DIGIT, BASE-A FORM
C
C               SIGN ( X(S-1)*A**(S-1) + ... + X(1)*A + X(0) )
C
C               WHERE 0 .LE. X(I) .LT. A FOR I=0,...,S-1.
C
C    I1MACH( 7) = A, THE BASE.
C
C    I1MACH( 8) = S, THE NUMBER OF BASE-A DIGITS.
C
C    I1MACH( 9) = A**S - 1, THE LARGEST MAGNITUDE.
C
C  FLOATING-POINT NUMBERS.
C
C    ASSUME FLOATING-POINT NUMBERS ARE REPRESENTED IN THE T-DIGIT,
C    BASE-B FORM
C
C               SIGN (B**E)*( (X(1)/B) + ... + (X(T)/B**T) )
C
C               WHERE 0 .LE. X(I) .LT. B FOR I=1,...,T,
C               0 .LT. X(1), AND EMIN .LE. E .LE. EMAX.
C
C    I1MACH(10) = B, THE BASE.
C
C  SINGLE-PRECISION
C
C    I1MACH(11) = T, THE NUMBER OF BASE-B DIGITS.
C
C    I1MACH(12) = EMIN, THE SMALLEST EXPONENT E.
C
C    I1MACH(13) = EMAX, THE LARGEST EXPONENT E.
C
C  DOUBLE-PRECISION
C
C    I1MACH(14) = T, THE NUMBER OF BASE-B DIGITS.
C
C    I1MACH(15) = EMIN, THE SMALLEST EXPONENT E.
C
C    I1MACH(16) = EMAX, THE LARGEST EXPONENT E.
C
C  TO ALTER THIS FUNCTION FOR A PARTICULAR ENVIRONMENT,
C  THE DESIRED SET OF DATA STATEMENTS SHOULD BE ACTIVATED BY
C  REMOVING THE C FROM COLUMN 1.  ALSO, THE VALUES OF
C  I1MACH(1) - I1MACH(4) SHOULD BE CHECKED FOR CONSISTENCY
C  WITH THE LOCAL OPERATING SYSTEM.
C  ON RARE MACHINES A STATIC STATEMENT MAY NEED TO BE ADDED.
C  (BUT PROBABLY MORE SYSTEMS PROHIBIT IT THAN REQUIRE IT.)
C
      INTEGER IMACH(16),OUTPUT
C
      EQUIVALENCE (IMACH(4),OUTPUT)
C
C     MACHINE CONSTANTS FOR THE BURROUGHS 1700 SYSTEM.
C
C      DATA IMACH( 1) /    7 /
C      DATA IMACH( 2) /    2 /
C      DATA IMACH( 3) /    2 /
C      DATA IMACH( 4) /    2 /
C      DATA IMACH( 5) /   36 /
C      DATA IMACH( 6) /    4 /
C      DATA IMACH( 7) /    2 /
C      DATA IMACH( 8) /   33 /
C      DATA IMACH( 9) / Z1FFFFFFFF /
C      DATA IMACH(10) /    2 /
C      DATA IMACH(11) /   24 /
C      DATA IMACH(12) / -256 /
C      DATA IMACH(13) /  255 /
C      DATA IMACH(14) /   60 /
C      DATA IMACH(15) / -256 /
C      DATA IMACH(16) /  255 /
C
C     MACHINE CONSTANTS FOR THE BURROUGHS 5700 SYSTEM.
C
C      DATA IMACH( 1) /   5 /
C      DATA IMACH( 2) /   6 /
C      DATA IMACH( 3) /   7 /
C      DATA IMACH( 4) /   6 /
C      DATA IMACH( 5) /  48 /
C      DATA IMACH( 6) /   6 /
C      DATA IMACH( 7) /   2 /
C      DATA IMACH( 8) /  39 /
C      DATA IMACH( 9) / O0007777777777777 /
C      DATA IMACH(10) /   8 /
C      DATA IMACH(11) /  13 /
C      DATA IMACH(12) / -50 /
C      DATA IMACH(13) /  76 /
C      DATA IMACH(14) /  26 /
C      DATA IMACH(15) / -50 /
C      DATA IMACH(16) /  76 /
C
C     MACHINE CONSTANTS FOR THE BURROUGHS 6700/7700 SYSTEMS.
C
C      DATA IMACH( 1) /   5 /
C      DATA IMACH( 2) /   6 /
C      DATA IMACH( 3) /   7 /
C      DATA IMACH( 4) /   6 /
C      DATA IMACH( 5) /  48 /
C      DATA IMACH( 6) /   6 /
C      DATA IMACH( 7) /   2 /
C      DATA IMACH( 8) /  39 /
C      DATA IMACH( 9) / O0007777777777777 /
C      DATA IMACH(10) /   8 /
C      DATA IMACH(11) /  13 /
C      DATA IMACH(12) / -50 /
C      DATA IMACH(13) /  76 /
C      DATA IMACH(14) /  26 /
C      DATA IMACH(15) / -32754 /
C      DATA IMACH(16) /  32780 /
C
C     MACHINE CONSTANTS FOR THE CDC 6000/7000 SERIES.
C
C      DATA IMACH( 1) /    5 /
C      DATA IMACH( 2) /    6 /
C      DATA IMACH( 3) /    7 /
C      DATA IMACH( 4) /    6 /
C      DATA IMACH( 5) /   60 /
C      DATA IMACH( 6) /   10 /
C      DATA IMACH( 7) /    2 /
C      DATA IMACH( 8) /   48 /
C      DATA IMACH( 9) / 00007777777777777777B /
C      DATA IMACH(10) /    2 /
C      DATA IMACH(11) /   48 /
C      DATA IMACH(12) / -974 /
C      DATA IMACH(13) / 1070 /
C      DATA IMACH(14) /   96 /
C      DATA IMACH(15) / -927 /
C      DATA IMACH(16) / 1070 /
C
C     MACHINE CONSTANTS FOR CONVEX C-1
C
C      DATA IMACH( 1) /    5 /
C      DATA IMACH( 2) /    6 /
C      DATA IMACH( 3) /    7 /
C      DATA IMACH( 4) /    6 /
C      DATA IMACH( 5) /   32 /
C      DATA IMACH( 6) /    4 /
C      DATA IMACH( 7) /    2 /
C      DATA IMACH( 8) /   31 /
C      DATA IMACH( 9) / 2147483647 /
C      DATA IMACH(10) /    2 /
C      DATA IMACH(11) /   24 /
C      DATA IMACH(12) / -128 /
C      DATA IMACH(13) /  127 /
C      DATA IMACH(14) /   53 /
C      DATA IMACH(15) /-1024 /
C      DATA IMACH(16) / 1023 /
C
C     MACHINE CONSTANTS FOR THE CRAY 1
C
C      DATA IMACH( 1) /     5 /
C      DATA IMACH( 2) /     6 /
C      DATA IMACH( 3) /   102 /
C      DATA IMACH( 4) /     6 /
C      DATA IMACH( 5) /    64 /
C      DATA IMACH( 6) /     8 /
C      DATA IMACH( 7) /     2 /
C      DATA IMACH( 8) /    46 /
C      DATA IMACH( 9) /  1777777777777777B /
C      DATA IMACH(10) /     2 /
C      DATA IMACH(11) /    47 /
C      DATA IMACH(12) / -8189 /
C      DATA IMACH(13) /  8190 /
C      DATA IMACH(14) /    94 /
C      DATA IMACH(15) / -8099 /
C      DATA IMACH(16) /  8190 /
C
C     MACHINE CONSTANTS FOR THE DATA GENERAL ECLIPSE S/200
C
C      DATA IMACH( 1) /   11 /
C      DATA IMACH( 2) /   12 /
C      DATA IMACH( 3) /    8 /
C      DATA IMACH( 4) /   10 /
C      DATA IMACH( 5) /   16 /
C      DATA IMACH( 6) /    2 /
C      DATA IMACH( 7) /    2 /
C      DATA IMACH( 8) /   15 /
C      DATA IMACH( 9) /32767 /
C      DATA IMACH(10) /   16 /
C      DATA IMACH(11) /    6 /
C      DATA IMACH(12) /  -64 /
C      DATA IMACH(13) /   63 /
C      DATA IMACH(14) /   14 /
C      DATA IMACH(15) /  -64 /
C      DATA IMACH(16) /   63 /
C
C     MACHINE CONSTANTS FOR THE HARRIS SLASH 6 AND SLASH 7
C
C      DATA IMACH( 1) /       5 /
C      DATA IMACH( 2) /       6 /
C      DATA IMACH( 3) /       0 /
C      DATA IMACH( 4) /       6 /
C      DATA IMACH( 5) /      24 /
C      DATA IMACH( 6) /       3 /
C      DATA IMACH( 7) /       2 /
C      DATA IMACH( 8) /      23 /
C      DATA IMACH( 9) / 8388607 /
C      DATA IMACH(10) /       2 /
C      DATA IMACH(11) /      23 /
C      DATA IMACH(12) /    -127 /
C      DATA IMACH(13) /     127 /
C      DATA IMACH(14) /      38 /
C      DATA IMACH(15) /    -127 /
C      DATA IMACH(16) /     127 /
C
C     MACHINE CONSTANTS FOR THE HONEYWELL DPS 8/70 SERIES.
C
C      DATA IMACH( 1) /    5 /
C      DATA IMACH( 2) /    6 /
C      DATA IMACH( 3) /   43 /
C      DATA IMACH( 4) /    6 /
C      DATA IMACH( 5) /   36 /
C      DATA IMACH( 6) /    4 /
C      DATA IMACH( 7) /    2 /
C      DATA IMACH( 8) /   35 /
C      DATA IMACH( 9) / O377777777777 /
C      DATA IMACH(10) /    2 /
C      DATA IMACH(11) /   27 /
C      DATA IMACH(12) / -127 /
C      DATA IMACH(13) /  127 /
C      DATA IMACH(14) /   63 /
C      DATA IMACH(15) / -127 /
C      DATA IMACH(16) /  127 /
C
C     MACHINE CONSTANTS FOR IEEE ARITHMETIC MACHINES (E.G., AT&T 3B
C     SERIES COMPUTERS AND 8087-BASED MACHINES LIKE THE IBM PC).
C
      DATA IMACH( 1) /    5 /
      DATA IMACH( 2) /    6 /
      DATA IMACH( 3) /    7 /
      DATA IMACH( 4) /    6 /
      DATA IMACH( 5) /   32 /
      DATA IMACH( 6) /    4 /
      DATA IMACH( 7) /    2 /
      DATA IMACH( 8) /   31 /
      DATA IMACH( 9) / 2147483647 /
      DATA IMACH(10) /    2 /
      DATA IMACH(11) /   24 /
      DATA IMACH(12) / -125 /
      DATA IMACH(13) /  128 /
      DATA IMACH(14) /   53 /
      DATA IMACH(15) / -1021 /
      DATA IMACH(16) /  1024 /
C
C     MACHINE CONSTANTS FOR THE IBM 360/370 SERIES,
C     THE XEROX SIGMA 5/7/9 AND THE SEL SYSTEMS 85/86.
C
C      DATA IMACH( 1) /   5 /
C      DATA IMACH( 2) /   6 /
C      DATA IMACH( 3) /   7 /
C      DATA IMACH( 4) /   6 /
C      DATA IMACH( 5) /  32 /
C      DATA IMACH( 6) /   4 /
C      DATA IMACH( 7) /   2 /
C      DATA IMACH( 8) /  31 /
C      DATA IMACH( 9) / Z7FFFFFFF /
C      DATA IMACH(10) /  16 /
C      DATA IMACH(11) /   6 /
C      DATA IMACH(12) / -64 /
C      DATA IMACH(13) /  63 /
C      DATA IMACH(14) /  14 /
C      DATA IMACH(15) / -64 /
C      DATA IMACH(16) /  63 /
C
C     MACHINE CONSTANTS FOR THE INTERDATA 8/32
C     WITH THE UNIX SYSTEM FORTRAN 77 COMPILER.
C
C     FOR THE INTERDATA FORTRAN VII COMPILER REPLACE
C     THE Z'S SPECIFYING HEX CONSTANTS WITH Y'S.
C
C      DATA IMACH( 1) /   5 /
C      DATA IMACH( 2) /   6 /
C      DATA IMACH( 3) /   6 /
C      DATA IMACH( 4) /   6 /
C      DATA IMACH( 5) /  32 /
C      DATA IMACH( 6) /   4 /
C      DATA IMACH( 7) /   2 /
C      DATA IMACH( 8) /  31 /
C      DATA IMACH( 9) / Z'7FFFFFFF' /
C      DATA IMACH(10) /  16 /
C      DATA IMACH(11) /   6 /
C      DATA IMACH(12) / -64 /
C      DATA IMACH(13) /  62 /
C      DATA IMACH(14) /  14 /
C      DATA IMACH(15) / -64 /
C      DATA IMACH(16) /  62 /
C
C     MACHINE CONSTANTS FOR THE PDP-10 (KA PROCESSOR).
C
C      DATA IMACH( 1) /    5 /
C      DATA IMACH( 2) /    6 /
C      DATA IMACH( 3) /    7 /
C      DATA IMACH( 4) /    6 /
C      DATA IMACH( 5) /   36 /
C      DATA IMACH( 6) /    5 /
C      DATA IMACH( 7) /    2 /
C      DATA IMACH( 8) /   35 /
C      DATA IMACH( 9) / "377777777777 /
C      DATA IMACH(10) /    2 /
C      DATA IMACH(11) /   27 /
C      DATA IMACH(12) / -128 /
C      DATA IMACH(13) /  127 /
C      DATA IMACH(14) /   54 /
C      DATA IMACH(15) / -101 /
C      DATA IMACH(16) /  127 /
C
C     MACHINE CONSTANTS FOR THE PDP-10 (KI PROCESSOR).
C
C      DATA IMACH( 1) /    5 /
C      DATA IMACH( 2) /    6 /
C      DATA IMACH( 3) /    7 /
C      DATA IMACH( 4) /    6 /
C      DATA IMACH( 5) /   36 /
C      DATA IMACH( 6) /    5 /
C      DATA IMACH( 7) /    2 /
C      DATA IMACH( 8) /   35 /
C      DATA IMACH( 9) / "377777777777 /
C      DATA IMACH(10) /    2 /
C      DATA IMACH(11) /   27 /
C      DATA IMACH(12) / -128 /
C      DATA IMACH(13) /  127 /
C      DATA IMACH(14) /   62 /
C      DATA IMACH(15) / -128 /
C      DATA IMACH(16) /  127 /
C
C     MACHINE CONSTANTS FOR PDP-11 FORTRANS SUPPORTING
C     32-BIT INTEGER ARITHMETIC.
C
C      DATA IMACH( 1) /    5 /
C      DATA IMACH( 2) /    6 /
C      DATA IMACH( 3) /    7 /
C      DATA IMACH( 4) /    6 /
C      DATA IMACH( 5) /   32 /
C      DATA IMACH( 6) /    4 /
C      DATA IMACH( 7) /    2 /
C      DATA IMACH( 8) /   31 /
C      DATA IMACH( 9) / 2147483647 /
C      DATA IMACH(10) /    2 /
C      DATA IMACH(11) /   24 /
C      DATA IMACH(12) / -127 /
C      DATA IMACH(13) /  127 /
C      DATA IMACH(14) /   56 /
C      DATA IMACH(15) / -127 /
C      DATA IMACH(16) /  127 /
C
C     MACHINE CONSTANTS FOR PDP-11 FORTRANS SUPPORTING
C     16-BIT INTEGER ARITHMETIC.
C
C      DATA IMACH( 1) /    5 /
C      DATA IMACH( 2) /    6 /
C      DATA IMACH( 3) /    7 /
C      DATA IMACH( 4) /    6 /
C      DATA IMACH( 5) /   16 /
C      DATA IMACH( 6) /    2 /
C      DATA IMACH( 7) /    2 /
C      DATA IMACH( 8) /   15 /
C      DATA IMACH( 9) / 32767 /
C      DATA IMACH(10) /    2 /
C      DATA IMACH(11) /   24 /
C      DATA IMACH(12) / -127 /
C      DATA IMACH(13) /  127 /
C      DATA IMACH(14) /   56 /
C      DATA IMACH(15) / -127 /
C      DATA IMACH(16) /  127 /
C
C     MACHINE CONSTANTS FOR THE SUN MICROSYSTEMS UNIX F77 COMPILER.
C
C      DATA IMACH( 1) /     5 /
C      DATA IMACH( 2) /     6 /
C      DATA IMACH( 3) /     6 /
C      DATA IMACH( 4) /     0 /
C      DATA IMACH( 5) /    32 /
C      DATA IMACH( 6) /     4 /
C      DATA IMACH( 7) /     2 /
C      DATA IMACH( 8) /    32 /
C      DATA IMACH( 9) /2147483647/
C      DATA IMACH(10) /     2 /
C      DATA IMACH(11) /    24 /
C      DATA IMACH(12) /  -126 /
C      DATA IMACH(13) /   128 /
C      DATA IMACH(14) /    53 /
C      DATA IMACH(15) / -1022 /
C      DATA IMACH(16) /  1024 /
C
C     MACHINE CONSTANTS FOR THE ALLIANT FX/8 UNIX FORTRAN COMPILER.
C
C$$$      DATA IMACH( 1) /     5 /
C$$$      DATA IMACH( 2) /     6 /
C$$$      DATA IMACH( 3) /     6 /
C$$$      DATA IMACH( 4) /     0 /
C$$$      DATA IMACH( 5) /    32 /
C$$$      DATA IMACH( 6) /     4 /
C$$$      DATA IMACH( 7) /     2 /
C$$$      DATA IMACH( 8) /    32 /
C$$$      DATA IMACH( 9) /2147483647/
C$$$      DATA IMACH(10) /     2 /
C$$$      DATA IMACH(11) /    24 /
C$$$      DATA IMACH(12) /  -126 /
C$$$      DATA IMACH(13) /   128 /
C$$$      DATA IMACH(14) /    53 /
C$$$      DATA IMACH(15) / -1022 /
C$$$      DATA IMACH(16) /  1024 /
C
C     MACHINE CONSTANTS FOR THE ALLIANT FX/8 UNIX FORTRAN COMPILER.
C     WITH THE -r8 COMMAND LINE OPTION.
C
C$$$      DATA IMACH( 1) /     5 /
C$$$      DATA IMACH( 2) /     6 /
C$$$      DATA IMACH( 3) /     6 /
C$$$      DATA IMACH( 4) /     0 /
C$$$      DATA IMACH( 5) /    32 /
C$$$      DATA IMACH( 6) /     4 /
C$$$      DATA IMACH( 7) /     2 /
C$$$      DATA IMACH( 8) /    32 /
C$$$      DATA IMACH( 9) /2147483647/
C$$$      DATA IMACH(10) /     2 /
C$$$      DATA IMACH(11) /    53 /
C$$$      DATA IMACH(12) / -1022 /
C$$$      DATA IMACH(13) /  1024 /
C$$$      DATA IMACH(14) /    53 /
C$$$      DATA IMACH(15) / -1022 /
C$$$      DATA IMACH(16) /  1024 /
C
C     MACHINE CONSTANTS FOR THE UNIVAC 1100 SERIES.
C
C     NOTE THAT THE PUNCH UNIT, I1MACH(3), HAS BEEN SET TO 7
C     WHICH IS APPROPRIATE FOR THE UNIVAC-FOR SYSTEM.
C     IF YOU HAVE THE UNIVAC-FTN SYSTEM, SET IT TO 1.
C
C      DATA IMACH( 1) /    5 /
C      DATA IMACH( 2) /    6 /
C      DATA IMACH( 3) /    7 /
C      DATA IMACH( 4) /    6 /
C      DATA IMACH( 5) /   36 /
C      DATA IMACH( 6) /    6 /
C      DATA IMACH( 7) /    2 /
C      DATA IMACH( 8) /   35 /
C      DATA IMACH( 9) / O377777777777 /
C      DATA IMACH(10) /    2 /
C      DATA IMACH(11) /   27 /
C      DATA IMACH(12) / -128 /
C      DATA IMACH(13) /  127 /
C      DATA IMACH(14) /   60 /
C      DATA IMACH(15) /-1024 /
C      DATA IMACH(16) / 1023 /
C
C     MACHINE CONSTANTS FOR VAX
C
C      DATA IMACH( 1) /    5 /
C      DATA IMACH( 2) /    6 /
C      DATA IMACH( 3) /    7 /
C      DATA IMACH( 4) /    6 /
C      DATA IMACH( 5) /   32 /
C      DATA IMACH( 6) /    4 /
C      DATA IMACH( 7) /    2 /
C      DATA IMACH( 8) /   31 /
C      DATA IMACH( 9) / 2147483647 /
C      DATA IMACH(10) /    2 /
C      DATA IMACH(11) /   24 /
C      DATA IMACH(12) / -127 /
C      DATA IMACH(13) /  127 /
C      DATA IMACH(14) /   56 /
C      DATA IMACH(15) / -127 /
C      DATA IMACH(16) /  127 /
C
C     MACHINE CONSTANTS FOR THE SEQUENT BALANCE 8000 AND SVS FORTRAN ON
C     THE AT&T 7300 (UNIX PC)
C
C      DATA IMACH( 1) /     0 /
C      DATA IMACH( 2) /     0 /
C      DATA IMACH( 3) /     7 /
C      DATA IMACH( 4) /     0 /
C      DATA IMACH( 5) /    32 /
C      DATA IMACH( 6) /     1 /
C      DATA IMACH( 7) /     2 /
C      DATA IMACH( 8) /    31 /
C      DATA IMACH( 9) /  2147483647 /
C      DATA IMACH(10) /     2 /
C      DATA IMACH(11) /    24 /
C      DATA IMACH(12) /  -125 /
C      DATA IMACH(13) /   128 /
C      DATA IMACH(14) /    53 /
C      DATA IMACH(15) / -1021 /
C      DATA IMACH(16) /  1024 /
C
C     MACHINE CONSTANTS FOR THE RM FORTRAN ON THE AT&T 7300 (UNIX PC)
C
C      DATA IMACH( 1) /     5 /
C      DATA IMACH( 2) /     6 /
C      DATA IMACH( 3) /     7 /
C      DATA IMACH( 4) /     6 /
C      DATA IMACH( 5) /    32 /
C      DATA IMACH( 6) /     1 /
C      DATA IMACH( 7) /     2 /
C      DATA IMACH( 8) /    31 /
C      DATA IMACH( 9) /  2147483647 /
C      DATA IMACH(10) /     2 /
C      DATA IMACH(11) /    24 /
C      DATA IMACH(12) /  -125 /
C      DATA IMACH(13) /   128 /
C      DATA IMACH(14) /    53 /
C      DATA IMACH(15) / -1021 /
C      DATA IMACH(16) /  1024 /
C
      IF (I .LT. 1  .OR.  I .GT. 16) GO TO 999
      I1MACH=IMACH(I)
      RETURN
  999 WRITE(OUTPUT,1999) I
 1999 FORMAT(' I1MACH - I OUT OF BOUNDS',I10)
      STOP
      END

C
