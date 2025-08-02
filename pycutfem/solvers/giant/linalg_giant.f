C
C*    Group  Iterative linear solver Good Broyden
C
      SUBROUTINE GBIT1I(N,LITMAX,RTOL,XSCAL,MULJAC,PRECON,
     $                 X,DEL,RHS,IOPT,QSUCC,IERR,KMAX,DELTA,DELTAK,
     $                 Q,Z,SITER,T,SIGMA,
     $                 ITER,K,L,SIGH,SIGMAK,TAUN1,TAUN2,SOLNRA,
     $                 LIWKU,IWKU,LRWKU,RWKU,
     $                 DELNRM,SOLNRM,DIFNRM,RHOS,TAUMIN,TAUMAX,TAUEQU,
     $                 NMULJ, IDMUL, IDPRE)
C*    Begin Prologue GBIT1I
      INTEGER N,LITMAX
      DOUBLE PRECISION RTOL,XSCAL(N)
      EXTERNAL MULJAC,PRECON
      INTEGER IOPT(50)
      LOGICAL QSUCC
      INTEGER IERR,KMAX
      DOUBLE PRECISION X(N),DEL(N),RHS(N),DELTA(N,KMAX),DELTAK(N),
     $                 Q(N),Z(N),SITER(N),T(KMAX),
     $                 SIGMA(KMAX)
      INTEGER ITER,K
      DOUBLE PRECISION SIGH(L),SIGMAK,TAUN1,TAUN2,SOLNRA
      INTEGER LIWKU
      INTEGER IWKU(LIWKU)
      INTEGER LRWKU
      DOUBLE PRECISION RWKU(LRWKU)
      DOUBLE PRECISION DELNRM,SOLNRM,DIFNRM,RHOS,TAUMIN,TAUMAX,TAUEQU
      INTEGER NMULJ,IDMUL,IDPRE
C     ------------------------------------------------------------
C
C*  Title
C
C     Good Broyden -
C     Iterative solution of a linear system (implicitly given matrix
C     by subroutine MULJAC - computes A*vector)
C
C     Internal core subroutine of GBIT1
C
C*  Written by        U. Nowak, L. Weimann
C*  Purpose           Iterative solution of large scale systems of 
C                     linear equations
C*  Method            Secant method Good Broyden with adapted
C                     linesearch
C*  Category          D2a. - Large Systems of Linear Equations
C*  Keywords          Linear Equations; Large Systems;
C*                    Iterative Methods
C*  Version           1.1
C*  Revision          March 1991
C*  Latest Change     March 1991
C
C     ------------------------------------------------------------
C*    End Prologue
      DOUBLE PRECISION ZERO,ONE,GREAT
      INTEGER L
      PARAMETER( ZERO=0.0D0, ONE = 1.0D0, GREAT=1.0D+35 )
      EXTERNAL SPRODS
      INTRINSIC DABS,DSQRT
      INTEGER I,J,K1,IFINI,MPRMON,ISYM,LUMON,IPLOT,ITERM
      DOUBLE PRECISION DN,EPMACH,SMALL,FAKTOR,FAKT1,DIFLIM,
     $                 SPRODS,SIGMAP,GAMMA,TAU,TK,
     $                 SIGQN,DIFQN,
     $                 TKRMI,TKRMA
      DOUBLE PRECISION SIGNEW,SIGQ1,SIGMIN
      LOGICAL QSTOP,QSTOPK
C*    Begin
C
C  Initiation
C------------
C
      CALL ZIBCONST(EPMACH,SMALL)
      IERR=10000
      ISYM=0
      ITERM=0
C
      IFINI =IOPT(33)
      MPRMON=IOPT(13)
      LUMON =IOPT(14)
      IPROPT=IOPT(17)
      IPLOT =IOPT(22)
      MPRTIM=IOPT(19)
      TKRMI=ONE
      TKRMA=ONE
      SIGQN=GREAT
      DIFQN=GREAT
      DIFNRM=GREAT
      SOLNRM=GREAT
      TAU=GREAT
      DIFLIM=1.0D10
      QSTOP = .FALSE.
      QSTOPK = .FALSE.
      DN = DBLE(FLOAT(N))
      SIGMIN=DN*EPMACH**2
C
C  initial preparations
C----------------------
C
      DO 2 I=1,N
        SITER(I) = X(I) / XSCAL(I)
        X(I) = (X(I)-DEL(I)) / XSCAL(I)
        DEL(I) =DEL(I) / XSCAL(I)
2     CONTINUE
C
C     continuation entry
C
      IF (QSUCC) THEN
        SOLNRM=SOLNRA
        GOTO 20
      ENDIF
C
C   initiation for new iteration run
C   --------------------------------
C
      TAUN1=GREAT
      TAUN2=GREAT
      NMULJ=0
C
C  initial print
C
      IF (MPRMON.LT.0) THEN
        SOLNRM = DSQRT( SPRODS(N, X, X)/DN )
        IF (IPROPT.EQ.0) WRITE(LUMON,10020)
10020   FORMAT(2X,' It',7X,'Cor',7X,'Sol',1X,'EstAbsErr',6X,'SolQ',
     $         7X,'tau')
        IF (IPROPT.EQ.1) WRITE(LUMON,10030)
10030   FORMAT(2X,' It',7X,'Cor',7X,'Del',7X,'Sol',1X,'EstAbsErr',
     $           6X,'DelQ',6X,'SolQ',7X,'tau')
      ENDIF
      ITER = 0
      IOPT(49) = IOPT(49) + 1
C
C  start / entry for restart of iteration
C----------------------------------------
C
3     CONTINUE
C
C     k := 0 
      K = 0
C
C------------------------------------------------------------
C     --- r0 := b-A*x0   ===  z := rhs-A*siter ---
C     --- delta0 := H0*r0  ===  Solve A(precon)*deltak = z ---
C     --- sigma0 := delta0(T)*delta0 === sigmak := deltak(T)*deltak ---
C------------------------------------------------------------
C
      DO 8 I=1,N
        Z(I) = SITER(I)*XSCAL(I)
8     CONTINUE
      IF (MPRTIM.NE.0) CALL MONON (IDMUL)
      CALL MULJAC(N, Z, Q, RWKU, IWKU)
      IF (MPRTIM.NE.0) CALL MONOFF (IDMUL)
C
      DO 10 I=1,N
        Q(I) = RHS(I) - Q(I)
10    CONTINUE
C
      IF (MPRTIM.NE.0) CALL MONON (IDPRE)
      CALL PRECON(N, Q, DELTAK, RWKU, IWKU)
      IF (MPRTIM.NE.0) CALL MONOFF (IDPRE)
      NMULJ=NMULJ+1
      DO 15 I=1,N
        DELTAK(I) = DELTAK(I)/XSCAL(I)
15    CONTINUE
C 
      SIGMAK = SPRODS(N, DELTAK, DELTAK)
      IF (SIGMAK.LE.SIGMIN) GOTO 1000
C
      CALL ITZMID(L,ITER,SIGH,SIGMAK,SIGQN,.FALSE.)
C
C  Main iteration loop         
C---------------------
C
20    IF ( .NOT. QSTOPK .OR. .NOT. QSTOP ) THEN
C
        IF (ITER.GE.LITMAX) THEN
          IERR = -1
          GOTO 998 
        ENDIF
        K1=K+1
C
C------------------------------------------------------------
C    --- qk := A*deltak  ===  q := A*deltak ---
C    --- z0quer := H0*qk  ===  Solve A(precon)*z = q ---
C------------------------------------------------------------
C
        DO 30 I=1,N
          Z(I) = DELTAK(I)*XSCAL(I)
30      CONTINUE
        IF (MPRTIM.NE.0) CALL MONON (IDMUL)
        CALL MULJAC(N, Z, Q, RWKU, IWKU)
        IF (MPRTIM.NE.0) CALL MONOFF (IDMUL)
C
        IF (MPRTIM.NE.0) CALL MONON (IDPRE)
        CALL PRECON(N, Q, Z, RWKU, IWKU)
        IF (MPRTIM.NE.0) CALL MONOFF (IDPRE)
        NMULJ=NMULJ+1
        DO 40 I=1,N
          Z(I) = Z(I)/XSCAL(I)
40      CONTINUE
C
C  update loop
C------------- 
C
        DO 100 I=1,K-1
          FAKTOR = SPRODS(N, DELTA(1,I), Z) / SIGMA(I)
          FAKT1 = ONE-T(I)
          DO 110 J=1,N
            Z(J) = Z(J) + FAKTOR * (DELTA(J,I+1) - FAKT1*DELTA(J,I))
110       CONTINUE
100     CONTINUE
        IF (K.NE.0) THEN
          FAKTOR = SPRODS(N, DELTA(1,K), Z) / SIGMA(K)
          FAKT1 = ONE-T(K)
          DO 120 J=1,N
            Z(J) = Z(J) + FAKTOR * (DELTAK(J) - FAKT1*DELTA(J,K))
120       CONTINUE
        ENDIF
C
C----------------------------------------------------------
C    --- zk := zquerk ===  z now corresponds to zk ---
C    --- gammak := deltak(t)*zk ---
C    --- tauk := sigmak/gammak ---
C----------------------------------------------------------
C
        GAMMA = SPRODS(N, DELTAK, Z)
        IF (DABS(GAMMA).LE.SIGMIN) GOTO 1001
        IF (GAMMA.NE.ZERO) TAU = SIGMAK / GAMMA
        TK = TAU
C
C  check for restart condition
C-----------------------------
C
        IF (GAMMA.EQ.ZERO .OR. TAU.LT.TAUMIN .OR. TAU.GT.TAUMAX) THEN
          IF (K.EQ.1 .AND. GAMMA.NE.ZERO ) THEN
            IF ( DABS(TAUN2-TAU) .LT. TAUEQU*DABS(TAU) ) THEN
              IERR=-3
              GOTO 998
            ENDIF
            TAUN2 = TAUN1
            TAUN1 = TAU
          ENDIF
          IF(MPRMON.LT.0.AND.GAMMA.NE.ZERO)
     1       WRITE(LUMON,10003) TAU,ITER,K
10003     FORMAT(' >>> Restart required due to tauk = ',D10.3,
     $           '( iter=',I5,',k=',I3,' )')
          IF(MPRMON.LT.0.AND.GAMMA.EQ.ZERO) 
     $      WRITE(LUMON,*)' >>> Restart required due to gammak = 0.0d0'
          IF (K.EQ.0) THEN
            IF (GAMMA.EQ.ZERO) THEN
              IF (MPRMON.LT.0)  WRITE(LUMON,10013) 
10013         FORMAT(' >>> Termination - restart not possible')
              IERR = -4
              GOTO 998
            ENDIF
            IF (MPRMON.LT.0) WRITE(LUMON,10023) TAU,ITER,K
10023       FORMAT(' >>> Restart condition ignored - tau = ',D10.3,
     $             '( iter=',I5,',k=',I3,' )')
            IF (TAU.LT.TAUMIN) TK = TKRMI
            IF (TAU.GT.TAUMAX) TK = TKRMA
            IF (MPRMON.LT.0) WRITE(LUMON,10033) TK
10033       FORMAT (' >>> tk reset to ',D10.3)
          ELSE
            GOTO 9990
          ENDIF
        ENDIF
C
C----------------------------------------------------------
C    --- x(k+1) := xk + tk*deltak ---
C    --- delta(k+1) := deltak - tauk*zk (if tk=tauk) ---
C                        - or -
C    --- delta(k+1) := (1-tk+tauk)*deltak - tauk*zk (if tk<>tauk)
C    --- sigma(k+1) := delta(k+1)(t)*delta(k+1) ---
C----------------------------------------------------------
C
C  update iterate
C---------------- 
C
        DO 130 J=1,N
          DEL(J) = DEL(J) + TK*DELTAK(J)
          SITER(J) = X(J) + DEL(J)
130     CONTINUE
C
C  compute norms for converge check 
C
        SOLNRA = SOLNRM
        SOLNRM = DSQRT( SPRODS(N, SITER, SITER)/DN )
        DELNRM = DSQRT (SPRODS(N, DEL, DEL)/DN )
        DIFNRM = DABS(TK)*DSQRT(SIGMAK/DN)
C
C  save information to perform next update loop
C
        IF (K1.LE.KMAX) THEN
          DO 6 I=1,N
            DELTA(I,K1)=DELTAK(I)
6         CONTINUE
          SIGMA(K1) = SIGMAK
          T(K1) = TK
        ENDIF
C
C  new delta
C
         IF (TK.EQ.TAU) THEN
           DO 140 J=1,N
             DELTAK(J) = DELTAK(J) -  TAU * Z(J)
140        CONTINUE
         ELSE
           FAKTOR = ONE - TK + TAU
           DO 145 J=1,N
             DELTAK(J) = FAKTOR*DELTAK(J) -  TAU * Z(J)
145        CONTINUE
         ENDIF
C
C  new sigma
C
        SIGMAP = SIGMAK
        SIGMAK = SPRODS(N, DELTAK, DELTAK)
        SIGNEW = DSQRT(SIGMAK/DN)
C
Ctaum;  CALL ITZMID(L,ITER,SIGH,TK**2*SIGMAK,SIGQN,.TRUE.)
        CALL ITZMID(L,ITER,SIGH,SIGMAK,SIGQN,.TRUE.)
C
        SIGQ1=DSQRT(SIGQN/DN)
Ctaus;  SIGQ1=TK*DSQRT(SIGMAK/DN)
CSim;   SIGQ1=DSQRT(SIGMAK/DN)
C
C
C       write graphics data
C
        IF (IPLOT.GT.0) 
     $   WRITE(IPLOT,9510) ITER,K,DIFNRM,SIGNEW,SIGQ1,SIGQ1,SIGQ1,SOLNRM
9510    FORMAT(I4,I3,6(1PD12.3))
C
        DIFQN=RHOS*SIGQ1
C
C       --- Print monitor ---
C
        IF (MPRMON.LT.0) THEN
10004     FORMAT(1X,I4,7(1X,D9.2))
          IF (MOD(ITER,-MPRMON).EQ.0) THEN 
            IF (IPROPT.EQ.0)
     $          WRITE(LUMON,10004) ITER, DIFNRM, SOLNRA, DIFQN, SOLNRM,
     $                              TAU
            IF (IPROPT.EQ.1)
     $          WRITE(LUMON,10004) ITER, DIFNRM, DELNRM, SOLNRA,
     $                              DIFQN, DELNRM, SOLNRM, TAU 
          ENDIF 
        ENDIF
C
C  check for termination
C-----------------------
C  
        QSTOPK=QSTOP
C??!    QSTOPK=.TRUE.
        ITERM = 0
        IF (SIGMAK.LE.SIGMIN) QSTOPK=.TRUE. 
        IF (ITER.GE.L-1) THEN
CSim;   IF (ITER.GE.0) THEN
          IF (DIFQN.GT.GREAT*SOLNRM) IERR=-2
          IF (DIFQN.GT.DIFLIM) IERR=-2
          IF (IERR.EQ.-2) GOTO 998
          QSTOP = DIFQN .LE. RTOL*SOLNRM
          IF (QSTOP) ITERM = 1
          IF (IFINI.EQ.2)  QSTOP = DIFQN .LE. 0.25D0 * DELNRM .OR. QSTOP
          IF (ITERM.EQ.0 .AND. QSTOP) ITERM = 2
        ELSE
          QSTOP=.FALSE.
        ENDIF
        IF (SIGMAK.LE.SIGMIN) QSTOP=.TRUE. 
C 
C  decision for next step (proceed or restart due to kmax)
        K = K+1
        ITER = ITER+1
        IF (K.LE.KMAX) GOTO 20
C
C --- End of Iteration Loop ---
C
        IF (MPRMON.LT.0.AND.KMAX.GT.0) 
     $    WRITE(LUMON,10050) ITER
10050 FORMAT(' >>> Restart due to k > kmax ( iter=', I5, ' )')
C
C --- Restart ---
C
9990    CONTINUE
        IOPT(48) = IOPT(48) + (K*(K-5))/2 + 1 
        IOPT(49) = IOPT(49) + 1
        IF ( SIGMAK.GT.SIGMIN ) GOTO 3
      ENDIF
C
C  solution exit
C
1000  CONTINUE
      IF ( SIGMAK.GT.SIGMIN ) IOPT(48) = IOPT(48) + (K*(K-5))/2 + 1
      IERR = 0
      IF (MPRMON.LT.0)  WRITE(LUMON,*) ' > Solution found'
      GOTO 999
1001  CONTINUE
      IERR = 0
      IF (MPRMON.LT.0)  THEN
         WRITE(LUMON,*) ' > Solution found ???'
         WRITE(LUMON,*) ' sigmak/gamma:',SIGMAK,GAMMA
      ENDIF
      SIGMAK=GAMMA
      GOTO 999
C
C  emergency termination
C
998   CONTINUE
      IF (MPRMON.NE.0) THEN
        IF (IERR.EQ.-1) WRITE(LUMON,99801)  
99801   FORMAT(' >> Iter. number limit reached')
        IF (IERR.EQ.-2)  WRITE(LUMON,99802) 
99802   FORMAT(' >> Termination, since iteration diverges')
        IF (IERR.EQ.-3)  WRITE(LUMON,99803) TAUN1,TAU
99803   FORMAT(' >> Term., since bad tau oscillates, ',
     $         ' taui = ',D9.2,' , ',D9.2)
        IF (IERR.EQ.-4)  WRITE(LUMON,99804) 
99804   FORMAT(' >> Termination, since negative tau at restart')
      ENDIF
C
C  final update of return arguments
C
999   CONTINUE
      DO 9991 J=1,N
        X(J)=SITER(J)*XSCAL(J)
        DEL(J)=DEL(J)*XSCAL(J)
9991  CONTINUE
C
      IF (SIGMAK.LE.SIGMIN) THEN
        DIFNRM=SIGMAK
        RTOL=SIGMAK
        IOPT(50) = 0
      ELSE
        RTOL = DIFQN/SOLNRM
        DIFNRM = DABS(TAU)*DSQRT(SIGMAP/DN)
        IOPT(50) = ITERM
      ENDIF
      IF (ITER.NE.0) SOLNRA = SOLNRM
C
      IF (MPRMON.NE.0 .AND. IOPT(1).EQ.0)
     1   WRITE(LUMON,1090) ITER,RTOL,IOPT(50)
1090  FORMAT(10X,'LinSol:   ','Iter:',I6,'  EstPrec:',D11.2,
     1       '  TermCrit:',I2)
      IF (MPRMON.NE.0 .AND. IOPT(1).EQ.1)
     1   WRITE(LUMON,1091) ITER,RTOL,IOPT(50)
1091  FORMAT(10X,' Cont.:   ','Iter:',I6,'  EstPrec:',D11.2,
     1       '  TermCrit:',I2)
C
      RETURN
C
C  End of subroutine GBIT1I
C
      END
C
      SUBROUTINE ITZMID(L,ITER,DIFH,DIFNRM,DIFQN,QNEXT)
      INTEGER L,ITER
      DOUBLE PRECISION DIFH(L),DIFNRM,DIFQN
      LOGICAL QNEXT
      IF (ITER.LT.L) THEN
        DIFH(ITER+1) = DIFNRM
      ELSE
        IF (QNEXT) THEN
          DO 119 J=1,L-1
            DIFH(J)=DIFH(J+1)
119       CONTINUE
        ENDIF
        DIFH(L)=DIFNRM
      ENDIF
      IF (ITER.GE.L-1) THEN
        DIFQN = DIFH(1)+DIFH(L)
        DO 1191 J=2,L-1
          DIFQN = DIFQN+2.0D0*DIFH(J)
1191    CONTINUE
        DIFQN = DIFQN/DBLE(FLOAT(L+1))
      ENDIF
      RETURN
      END
C
      DOUBLE PRECISION FUNCTION SPRODS( N, X, Y )
      INTEGER N
      DOUBLE PRECISION X(N),Y(N)
      INTEGER I
      DOUBLE PRECISION S
C*    Begin
      S=0.0D0
      DO 10 I=1,N
        S = S + X(I)*Y(I)
10    CONTINUE
      SPRODS=S
      RETURN
      END
C
      SUBROUTINE DCOPY(N,DX,INCX,DY,INCY)
C
C     COPY DOUBLE PRECISION DX TO DOUBLE PRECISION DY.
C     FOR I = 0 TO N-1, COPY DX(LX+I*INCX) TO DY(LY+I*INCY),
C     WHERE LX = 1 IF INCX .GE. 0, ELSE LX = (-INCX)*N, AND LY IS
C     DEFINED IN A SIMILAR WAY USING INCY.
C
      DOUBLE PRECISION DX(1),DY(1)
      IF(N.LE.0)RETURN
      IF(INCX.EQ.INCY) IF(INCX-1) 5,20,60
    5 CONTINUE
C
C        CODE FOR UNEQUAL OR NONPOSITIVE INCREMENTS.
C
      IX = 1
      IY = 1
      IF(INCX.LT.0)IX = (-N+1)*INCX + 1
      IF(INCY.LT.0)IY = (-N+1)*INCY + 1
      DO 10 I = 1,N
        DY(IY) = DX(IX)
        IX = IX + INCX
        IY = IY + INCY
   10 CONTINUE
      RETURN
C
C        CODE FOR BOTH INCREMENTS EQUAL TO 1
C
C
C        CLEAN-UP LOOP SO REMAINING VECTOR LENGTH IS A MULTIPLE OF 7.
C
   20 M = MOD(N,7)
      IF( M .EQ. 0 ) GO TO 40
      DO 30 I = 1,M
        DY(I) = DX(I)
   30 CONTINUE
      IF( N .LT. 7 ) RETURN
   40 MP1 = M + 1
      DO 50 I = MP1,N,7
        DY(I) = DX(I)
        DY(I + 1) = DX(I + 1)
        DY(I + 2) = DX(I + 2)
        DY(I + 3) = DX(I + 3)
        DY(I + 4) = DX(I + 4)
        DY(I + 5) = DX(I + 5)
        DY(I + 6) = DX(I + 6)
   50 CONTINUE
      RETURN
C
C        CODE FOR EQUAL, POSITIVE, NONUNIT INCREMENTS.
C
   60 CONTINUE
      NS=N*INCX
          DO 70 I=1,NS,INCX
          DY(I) = DX(I)
   70     CONTINUE
      RETURN
      END
      DOUBLE PRECISION FUNCTION DNRM2 ( N, DX, INCX)
      INTEGER          NEXT
      DOUBLE PRECISION   DX(1), CUTLO, CUTHI, HITEST, SUM, XMAX,ZERO,ONE
      DATA   ZERO, ONE /0.0D0, 1.0D0/
C
C     EUCLIDEAN NORM OF THE N-VECTOR STORED IN DX() WITH STORAGE
C     INCREMENT INCX .
C     IF    N .LE. 0 RETURN WITH RESULT = 0.
C     IF N .GE. 1 THEN INCX MUST BE .GE. 1
C
C           C.L.LAWSON, 1978 JAN 08
C
C     FOUR PHASE METHOD     USING TWO BUILT-IN CONSTANTS THAT ARE
C     HOPEFULLY APPLICABLE TO ALL MACHINES.
C         CUTLO = MAXIMUM OF  DSQRT(U/EPS)  OVER ALL KNOWN MACHINES.
C         CUTHI = MINIMUM OF  DSQRT(V)      OVER ALL KNOWN MACHINES.
C     WHERE
C         EPS = SMALLEST NO. SUCH THAT EPS + 1. .GT. 1.
C         U   = SMALLEST POSITIVE NO.   (UNDERFLOW LIMIT)
C         V   = LARGEST  NO.            (OVERFLOW  LIMIT)
C
C     BRIEF OUTLINE OF ALGORITHM..
C
C     PHASE 1    SCANS ZERO COMPONENTS.
C     MOVE TO PHASE 2 WHEN A COMPONENT IS NONZERO AND .LE. CUTLO
C     MOVE TO PHASE 3 WHEN A COMPONENT IS .GT. CUTLO
C     MOVE TO PHASE 4 WHEN A COMPONENT IS .GE. CUTHI/M
C     WHERE M = N FOR X() REAL AND M = 2*N FOR COMPLEX.
C
C     VALUES FOR CUTLO AND CUTHI..
C     FROM THE ENVIRONMENTAL PARAMETERS LISTED IN THE IMSL CONVERTER
C     DOCUMENT THE LIMITING VALUES ARE AS FOLLOWS..
C     CUTLO, S.P.   U/EPS = 2**(-102) FOR  HONEYWELL.  CLOSE SECONDS ARE
C                   UNIVAC AND DEC AT 2**(-103)
C                   THUS CUTLO = 2**(-51) = 4.44089E-16
C     CUTHI, S.P.   V = 2**127 FOR UNIVAC, HONEYWELL, AND DEC.
C                   THUS CUTHI = 2**(63.5) = 1.30438E19
C     CUTLO, D.P.   U/EPS = 2**(-67) FOR HONEYWELL AND DEC.
C                   THUS CUTLO = 2**(-33.5) = 8.23181D-11
C     CUTHI, D.P.   SAME AS S.P.  CUTHI = 1.30438D19
C     DATA CUTLO, CUTHI / 8.232D-11,  1.304D19 /
C     DATA CUTLO, CUTHI / 4.441E-16,  1.304E19 /
      DATA CUTLO, CUTHI / 8.232D-11,  1.304D19 /
C
      IF(N .GT. 0) GO TO 10
         DNRM2  = ZERO
         GO TO 300
C
   10 ASSIGN 30 TO NEXT
      SUM = ZERO
      NN = N * INCX
C                                                 BEGIN MAIN LOOP
      I = 1
   20    GO TO NEXT,(30, 50, 70, 110)
   30 IF( DABS(DX(I)) .GT. CUTLO) GO TO 85
      ASSIGN 50 TO NEXT
      XMAX = ZERO
C
C                        PHASE 1.  SUM IS ZERO
C
   50 IF( DX(I) .EQ. ZERO) GO TO 200
      IF( DABS(DX(I)) .GT. CUTLO) GO TO 85
C
C                                PREPARE FOR PHASE 2.
      ASSIGN 70 TO NEXT
      GO TO 105
C
C                                PREPARE FOR PHASE 4.
C
  100 I = J
      ASSIGN 110 TO NEXT
      SUM = (SUM / DX(I)) / DX(I)
  105 XMAX = DABS(DX(I))
      GO TO 115
C
C                   PHASE 2.  SUM IS SMALL.
C                             SCALE TO AVOID DESTRUCTIVE UNDERFLOW.
C
   70 IF( DABS(DX(I)) .GT. CUTLO ) GO TO 75
C
C                     COMMON CODE FOR PHASES 2 AND 4.
C                     IN PHASE 4 SUM IS LARGE.  SCALE TO AVOID OVERFLOW.
C
  110 IF( DABS(DX(I)) .LE. XMAX ) GO TO 115
         SUM = ONE + SUM * (XMAX / DX(I))**2
         XMAX = DABS(DX(I))
         GO TO 200
C
  115 SUM = SUM + (DX(I)/XMAX)**2
      GO TO 200
C
C
C                  PREPARE FOR PHASE 3.
C
   75 SUM = (SUM * XMAX) * XMAX
C
C
C     FOR REAL OR D.P. SET HITEST = CUTHI/N
C     FOR COMPLEX      SET HITEST = CUTHI/(2*N)
C
   85 HITEST = CUTHI/FLOAT( N )
C
C                   PHASE 3.  SUM IS MID-RANGE.  NO SCALING.
C
      DO 95 J =I,NN,INCX
      IF(DABS(DX(J)) .GE. HITEST) GO TO 100
   95    SUM = SUM + DX(J)**2
      DNRM2 = DSQRT( SUM )
      GO TO 300
C
  200 CONTINUE
      I = I + INCX
      IF ( I .LE. NN ) GO TO 20
C
C              END OF MAIN LOOP.
C
C              COMPUTE SQUARE ROOT AND ADJUST FOR SCALING.
C
      DNRM2 = XMAX * DSQRT(SUM)
  300 CONTINUE
      RETURN
      END
