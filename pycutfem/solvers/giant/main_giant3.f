      PROGRAM MAINV3
C
C     Example program for usage of GIANT in connection with
C     SLAP linear algebra subroutines and alternative linear solver
C     GMRES (from SLAP package, slightly modified termination criterium
C     and code lines added for usage with time monitor package).
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      PARAMETER (MN=961, MNPDE=1, LRESER=50, KMAX=9)
C     parameter (lrwk=(kmax+14)*mn+kmax*(kmax+3)+71 , liwk=70)
      PARAMETER (LRWK=22282, LIWK=70)
C     parameter(mnzmax=5*mn, mnx=31, mny=31)
C     parameter(liwku=10+2*mnzmax+4*mn+3+11, lrwku=2*mnzmax+mnx+mny+2)
      PARAMETER(LIWKU=13478, LRWKU=9674)
      PARAMETER (IUMON=6)
C     parameter (lugout=78)
      PARAMETER (LUGOUT=0)
      DOUBLE PRECISION UH(MNPDE)
      DOUBLE PRECISION U(MN),USCAL(MN)
      DOUBLE PRECISION RWK(LRWK) 
      INTEGER IWK(LIWK) 
      DIMENSION IOPT(LRESER)
      INTEGER IWKU(LIWKU)
      DOUBLE PRECISION RWKU(LRWKU)
C
      EXTERNAL EFCN,EJAC,DSMV,SLPREC,ITSOL
      CHARACTER*100 FILNAM
      CHARACTER*16 CID
      DOUBLE PRECISION RPAR1, RPAR2
      INTEGER IPAR
C
C 
      IWKU(1)=LRWKU
      IWKU(2)=LIWKU
C
      IF (IUMON.NE.6) THEN
        WRITE(6,*) 'Output filename ?'
        READ(5,*) FILNAM
        OPEN(IUMON,FILE=FILNAM)
      ENDIF
C
      CID='testproblem     '
C 
C  dimension and domain of example
C
      NPDE=1
C
      XMIN=-3.D0
      XMAX=3.D0
C 
      YMIN=-3.D0
      YMAX=3.0D0
      RPAR1=1.0D0
      RPAR2=1.0D0
      IPAR=2
C
      IWKU(11)=NPDE
C
      WRITE(6,*) ' domain:'
      WRITE(6,*) XMIN,XMAX
      WRITE(6,*) YMIN,YMAX
C
C     number of grid points
C
      NX=31
      NY=31
      IWKU(12)=NX
      IWKU(13)=NY
      IWKU(3) = NX+NY+2
      IWKU(4) = 11
      NRWU=IWKU(3)
      IF (LRWKU.LT.NRWU) THEN
        WRITE(6,10002) NRWU
10002   FORMAT(' User function real workspace too small',/,
     $         ' at least required: ',I7)
        STOP
      ENDIF
      IWKU(21) = IPAR
      RWKU(NX+NY+1) = RPAR1
      RWKU(NX+NY+2) = RPAR2
C
C     construct aequidistant rectangular grid
C
      DX=(XMAX-XMIN)/DBLE(FLOAT(NX-1))
C
      DO 10 IX=1,NX
      RWKU(IX)=XMIN+DBLE(FLOAT(IX-1))*DX
10    CONTINUE
C
      DY=(YMAX-YMIN)/DBLE(FLOAT(NY-1))
      DO 20 IY=1,NY
      RWKU(NX+IY)=YMIN+DBLE(FLOAT(IY-1))*DY
20    CONTINUE
C
C
      N=NX*NY*NPDE
      NZMAX = (NPDE+4)*N
      IWKU(5)=NZMAX
      IWKU(14)=NZMAX
C
      WRITE(6,*) ' grid:'
      WRITE(6,*) (RWKU(I),I=1,NX)
      WRITE(6,*) (RWKU(NX+I),I=1,NY)
C 
C  set starting values for Newton iteration
C
      DO 100 IY=1,NY
        LY=(IY-1)*NX*NPDE
        Y=RWKU(NX+IY)
        DO 100 IX=1,NX
          L=LY+(IX-1)*NPDE
          X=RWKU(IX)
          Q=X**2/RPAR1**2 + Y**2/RPAR2**2
          UH(1)=0.2D0*DEXP(-Q)
          DO 100 K=1,NPDE
            U(L+K)=UH(K)
100   CONTINUE
C
C     iwku(9)=type of preconditioner to be used
C       0 = Incomplete LU Decomposition  Preconditioner SLAP
C       1 = Lower Triangle Preconditioner SLAP
C       2 = Diagonal Scaling Preconditioner SLAP
C       3 = None
C       4 = Block diagonal Scaling Preconditioner ZIB
C     
5     CONTINUE
C
      DO 110 I=1,LRESER
        IOPT(I)= 0
        IWK(I) = 0
        RWK(I) = 0.0D0
110   CONTINUE
C
C     linear solver : User solver (GMRES)
      IOPT(8)=9
C
C     Maximum number of linear solver iterations
      IWK(41)=3000
C
C     time monitor on/off and it's output unit
      IOPT(19)=1
      IOPT(20)=IUMON
C
C     default input values for Print Mon., PRECON, It.Sol., Newton It.
C 
C     iopt(2)=0 : standard ; =1 : one step mode
      IOPT(2)=0
C       Set MPRERR, MPRMON, MPRLIN (iwku(8) for SLAPInt)
      IOPT(11)=3
      IOPT(13)=3
      IOPT(17)=1
C     iopt(17)=1  : print only linear solvers messages
C     iopt(17)=-j : print each j-th iterate
      IWKU(8)=3
C       Set print units LUERR, LUMON, LULIN (iwku(7) for SLAPInt)
      IOPT(12)=IUMON
      IOPT(14)=IUMON
      IOPT(18)=IUMON
      IWKU(7)=IOPT(12)
C     Set preconditioner
      IWKU(9)=0
C     Set It. lin. sol. parameters
      IOPT(41)=KMAX
      RWK(41)=4.0D2
      RWK(42)=4.0D2
C     Set Newton alg. parameters
      IOPT(31)=3
      RTOL=1.0D-5
      IWK(31)=50
C
C     miscellaneous other options
C
C     scaling: 0 = internal scaling , 1 = user-only scaling
      IOPT(9)=0
C     Summary plot information unit
      IWK(21)=LUGOUT
C
C     print example characteristics
C
      WRITE (6,10001) IOPT(13),IOPT(17),  IWKU(9),
     $                IOPT(41), RWK(41), RWK(42),
     $                IOPT(31), RTOL, IWK(31)
10001 FORMAT(/, ' ============= Parameter settings ============= ',//,
     $       /,1X,'mprmon=',I1,'  mprlin=',I4,
     $       /,1X,'preconditioner:',I1,
     $       /,1X,'iterative linear solver : GMRES',
     $       /,1X,'kmax=',I3,'  rhoord=',D8.1,'  rhosim=',D8.1,
     $       /,1X,'nonlin=',I1,'  rtol=',D8.1,'  itmax=',I4,
     $       //,' ============================================== ',//)
C     size of blocks for block diagonal scaling
      IF (IWKU(9).EQ.4) IWKU(10)=NPDE
C
C
C     User scaling vector uscal
      DO 1 J=1,N
        USCAL(J)=1.0D0
1     CONTINUE
      IF (LUGOUT.NE.0) THEN
        L1=INDEX(CID,' ')
        IF (L1.EQ.0) L1=LEN(CID)+1
        FILNAM=CID(1:L1-1)//'.sum'
        OPEN(LUGOUT,FILE=FILNAM)
      ENDIF
C
      IERR=-1
      I=0
C
C     one-step mode loop starts here
C     (executed exactly one times, if one-step mode not selected)
C
31    IF (IERR.EQ.-1) THEN
        CALL GIANT(N,EFCN,EJAC,U,USCAL,RTOL,IOPT,IERR,
     $              LIWK,IWK,LRWK,RWK,
     $              LIWKU,IWKU,LRWKU,RWKU,DSMV,SLPREC,ITSOL)
C       Clear workspace declared not to be used
        NIFREE=IWK(16)
        DO 311 K=NIFREE,LIWK
          IWK(K)=0
311     CONTINUE
        NRFREE=IWK(17)
        DO 312 K=NRFREE,LRWK
          RWK(K)=0.0D0
312     CONTINUE
        I=I+1
32      FORMAT(' returned from call ',I4,' of GIANT')
        WRITE(6,32)I
C       IOPT(2)=0
        GOTO 31
      ENDIF
      CLOSE(2)
C
      IF (IUMON.NE.6) CLOSE(IUMON)
C 
      STOP
      END
C
      SUBROUTINE  FCN1(N,U,URHS,RWKU,IWKU,NFCN,IFAIL)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INTEGER N, IFAIL
      DOUBLE PRECISION U(N), URHS(N)
      DOUBLE PRECISION RWKU(*)
      INTEGER IWKU(*)
      NPDE = IWKU(1)
      NX   = IWKU(2)
      NY   = IWKU(3)
      IF (NFCN.EQ.0) THEN
        IWKU(6)=0
      ELSE
        IWKU(6)=IWKU(6)+1
      ENDIF
      T=0.0D0
      CALL FCN0 (N,NPDE,NX,NY,RWKU(1),RWKU(NX+1),RWKU(NX+NY+1),
     $           RWKU(NX+NY+2),IWKU(11),U,URHS)
      IFAIL=0
      RETURN
      END
C
      SUBROUTINE FCN0 (N,NPDE,NX,NY,XGRI,YGRI,RPAR1,RPAR2,IPAR,U,URHS)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DOUBLE PRECISION U(NPDE,NX,NY),URHS(NPDE,NX,NY)
      DOUBLE PRECISION XGRI(NX),YGRI(NY)
      PARAMETER (MNPDE=1) 
      DIMENSION UXX(MNPDE),UYY(MNPDE),UX(MNPDE),UY(MNPDE)
      DIMENSION DX(MNPDE),DY(MNPDE),CXY(MNPDE),G(MNPDE)
C 
C  interior nodes
C 
      DO 2000 IY=2,NY-1
       YL=YGRI(IY-1)
       YM=YGRI(IY)
       YR=YGRI(IY+1)
       DYL=YM-YL
       DYR=YR-YM
       DYHH=2.D0/((DYL+DYR)*DYL*DYR)
       DO 1000 IX=2,NX-1
        XL=XGRI(IX-1)
        XM=XGRI(IX)
        XR=XGRI(IX+1)
        DXL=XM-XL
        DXR=XR-XM
        DXHH=2.D0/((DXL+DXR)*DXL*DXR)
C 
C  approx. uxx and ux
C
        DO 114 J=1,NPDE
          UXX(J)=(DXL*U(J,IX+1,IY)-(DXL+DXR)*U(J,IX,IY)
     $           +DXR*U(J,IX-1,IY))*DXHH
          UX(J)=(DXL*(U(J,IX+1,IY)-U(J,IX,IY))/DXR
     $          +DXR*(U(J,IX,IY)-U(J,IX-1,IY))/DXL)/(DXL+DXR)
114     CONTINUE
C 
C  approx. uyy and uy
C
        DO 119 J=1,NPDE
          UYY(J)=(DYL*U(J,IX,IY+1)-(DYL+DYR)*U(J,IX,IY)
     $           +DYR*U(J,IX,IY-1))*DYHH
          UY(J)=(DYL*(U(J,IX,IY+1)-U(J,IX,IY))/DYR
     $          +DYR*(U(J,IX,IY)-U(J,IX,IY-1))/DYL)/(DYL+DYR)
119     CONTINUE
C 
C 
C  get terms
        CALL FUNS (NPDE,XM,YM,U(1,IX,IY),UX,UY,RPAR1,RPAR2,IPAR,
     $             DX,DY,CXY,G)
C
C
C  set up rhs at actual node
C
        DO 200 J=1,NPDE
          URHS(J,IX,IY)=DX(J)*UXX(J)+DY(J)*UYY(J)+CXY(J)+G(J)
200     CONTINUE
C
1000   CONTINUE
C
2000  CONTINUE
C 
C  boundary nodes 
C 
C     homogeneous dirichlet boundary conditions
C
C  x=xmin
      DO 3100 IY=1,NY
        DO 3100 J=1,NPDE
          URHS(J,1,IY)=U(J,1,IY)
3100  CONTINUE
C 
C  x=xmax
      DO 3200 IY=1,NY
        DO 3200 J=1,NPDE
          URHS(J,NX,IY)=U(J,NX,IY)
3200  CONTINUE
C
C  y=ymin
      DO 3300 IX=1,NX
        DO 3300 J=1,NPDE
          URHS(J,IX,1)=U(J,IX,1)
3300  CONTINUE
C
C  y=ymax
      DO 3400 IX=1,NX
        DO 3400 J=1,NPDE
          URHS(J,IX,NY)=U(J,IX,NY)
3400  CONTINUE
C
      RETURN
C
C  end subroutine fcn0
C
      END
C
      SUBROUTINE JAC1(FCN,N,U,UWGT,F,NZMAX,IDUMMY,A,IA,JA,NFILL,
     $                RWKU,IWKU,NJAC,IFAIL)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      EXTERNAL FCN
      INTEGER N
      DOUBLE PRECISION U(N),UWGT(N),F(N)
      INTEGER NZMAX,IDUMMY
      DOUBLE PRECISION A(NZMAX)
      INTEGER IA(NZMAX),JA(NZMAX)
      INTEGER NFILL
      DOUBLE PRECISION RWKU(*)
      INTEGER IWKU(*)
      INTEGER NJAC,IFAIL 
C
      NPDE = IWKU(1)
      NX   = IWKU(2)
      NY   = IWKU(3)
      CALL JACI( N, NPDE, NX, NY, RWKU(1), RWKU(NX+1),
     $           RWKU(NX+NY+1),RWKU(NX+NY+2),IWKU(11),
     $           U, NZMAX, A, IA, JA, NFILL, IFAIL)
      RETURN
      END
C
      SUBROUTINE JACI( N, NPDE, NX, NY, XGRI, YGRI, RPAR1, RPAR2, IPAR,
     $                 U, NZMAX, A, IA, JA, NFILL, IERR)
C
C 
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DOUBLE PRECISION U(NPDE,NX,NY)
      DOUBLE PRECISION XGRI(NX),YGRI(NY)
      INTEGER  NZMAX
      DOUBLE PRECISION A(NZMAX)
      INTEGER IA(NZMAX),JA(NZMAX)
      INTEGER NFILL
      PARAMETER (MNPDE=1) 
      DIMENSION UX(MNPDE),UY(MNPDE)
      DIMENSION DX(MNPDE),DY(MNPDE)
      DIMENSION DCDU(MNPDE,MNPDE),DCDXU(MNPDE,MNPDE),DCDYU(MNPDE,MNPDE),
     $          DGDU(MNPDE,MNPDE)
C
      IERR = 0
      NFILL = 0 
C 
C  interior nodes
C 
      DO 2000 IY=2,NY-1
       YL=YGRI(IY-1)
       YM=YGRI(IY)
       YR=YGRI(IY+1)
       DYL=YM-YL
       DYR=YR-YM
       DYHH=2.D0/((DYL+DYR)*DYL*DYR)
       UYYFAC=-2.0D0/(DYL*DYR)
       UYFAC=(DYR-DYL)/(DYL*DYR)
       UYYLFC=2.0D0/((DYL+DYR)*DYL)
       UYYRFC=2.0D0/((DYL+DYR)*DYR)
       UYLFAC=-(DYR/DYL)/(DYL+DYR)
       UYRFAC=(DYL/DYR)/(DYL+DYR)
       DO 1000 IX=2,NX-1
        XL=XGRI(IX-1)
        XM=XGRI(IX)
        XR=XGRI(IX+1)
        DXL=XM-XL
        DXR=XR-XM
        DXHH=2.D0/((DXL+DXR)*DXL*DXR)
        UXXFAC=-2.0D0/(DXL*DXR)
        UXFAC=(DXR-DXL)/(DXL*DXR)
        UXXLFC=2.0D0/((DXL+DXR)*DXL)
        UXXRFC=2.0D0/((DXL+DXR)*DXR)
        UXLFAC=-(DXR/DXL)/(DXL+DXR)
        UXRFAC=(DXL/DXR)/(DXL+DXR)
C
        IF (NFILL.GE.NZMAX) GOTO 9998
C 
C  approx. ux
C
        DO 114 J=1,NPDE
          UX(J)=(DXL*(U(J,IX+1,IY)-U(J,IX,IY))/DXR
     $           +DXR*(U(J,IX,IY)-U(J,IX-1,IY))/DXL)/(DXL+DXR)
114     CONTINUE
C 
C  approx. uy
C
        DO 119 J=1,NPDE
          UY(J)=(DYL*(U(J,IX,IY+1)-U(J,IX,IY))/DYR
     $           +DYR*(U(J,IX,IY)-U(J,IX,IY-1))/DYL)/(DYL+DYR)
119     CONTINUE
C 
C  get terms
C
        CALL DFUNS (NPDE,XM,YM,U(1,IX,IY),UX,UY,RPAR1,RPAR2,IPAR,
     $            DX,DY,DCDU,DCDXU,DCDYU,DGDU)
C
C  set up Jacobian elements at actual node
C
        IH=((IY-1)*NX+IX-1)*NPDE
        DO 200 J=1,NPDE
C
          NFILL = NFILL+1
          A(NFILL) = DCDYU(J,J)*UYLFAC + DY(J)*UYYLFC
          IA(NFILL) = IH+J
          JA(NFILL) = IH+J - NX*NPDE
C
          NFILL = NFILL+1
          A(NFILL) = DCDXU(J,J)*UXLFAC + DX(J)*UXXLFC
          IA(NFILL) = IH+J
          JA(NFILL) = IH+J - NPDE
C
          DO 230 K=1,NPDE
            NFILL = NFILL+1
            A(NFILL) = DCDU(J,K) + DCDXU(J,K)*UXFAC 
     $                           + DCDYU(J,K)*UYFAC + DGDU(J,K)
            IF (K.EQ.J) 
     $        A(NFILL) = A(NFILL) + DX(J)*UXXFAC+DY(J)*UYYFAC
            IA(NFILL) = IH+J
            JA(NFILL) = IH+K
230       CONTINUE
C
          NFILL = NFILL+1
          A(NFILL) = DCDXU(J,J)*UXRFAC + DX(J)*UXXRFC
          IA(NFILL) = IH+J
          JA(NFILL) = IH+J + NPDE
C
          NFILL = NFILL+1
          A(NFILL) = DCDYU(J,J)*UYRFAC + DY(J)*UYYRFC
          IA(NFILL) = IH+J
          JA(NFILL) = IH+J + NX*NPDE
200     CONTINUE
C
1000   CONTINUE
C
2000  CONTINUE
C 
C     boundary nodes
C 
C     homogeneous dirichlet boundary conditions
C
C  y=ymin
      DO 3300 IX=1,NX
        IXA = (IX-1)*NPDE
        DO 3300 J=1,NPDE
          NFILL = NFILL+1
          A (NFILL) = 1.0D0
          IA(NFILL) = IXA+J
          JA(NFILL) = IXA+J
3300  CONTINUE
C
C  x=xmin
      DO 3100 IY=2,NY-1
        IYA = (IY-1)*NX*NPDE
        DO 3100 J=1,NPDE
          NFILL = NFILL+1
          A (NFILL) = 1.0D0
          IA(NFILL) = IYA+J
          JA(NFILL) = IYA+J
3100  CONTINUE
C
C  x=xmax
      DO 3200 IY=2,NY-1
        IYA = (IY*NX-1)*NPDE
        DO 3200 J=1,NPDE
          NFILL = NFILL+1
          A (NFILL) = 1.0D0
          IA(NFILL) = IYA+J
          JA(NFILL) = IYA+J
3200  CONTINUE
C
C  y=ymax
      DO 3400 IX=1,NX
        IXA = ((IX-1)+(NY-1)*NX)*NPDE
        DO 3400 J=1,NPDE
          NFILL = NFILL+1
          A (NFILL) = 1.0D0
          IA(NFILL) = IXA+J
          JA(NFILL) = IXA+J
3400  CONTINUE
C
      GOTO 9999
9998  IERR=-1
9999  RETURN
      END
C
      SUBROUTINE FUNS (NPDE,X,Y,U,UX,UY,RPAR1,RPAR2,IPAR,DX,DY,CXY,G)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      PARAMETER(MNPDE=1)
C 
      DIMENSION U(NPDE),UX(NPDE),UY(NPDE),CXY(NPDE),G(NPDE)
      DIMENSION DX(NPDE),DY(NPDE),UH(MNPDE)
C 
C
         Q=X**2/RPAR1**2 + Y**2/RPAR2**2
         UH(1)=0.9D0*DEXP(-Q)+0.1D0*U(1)
         UXXH=UH(1)*(4.D0*X**2/RPAR1**4 - 2.D0/RPAR1**2)
         UXH=-UH(1)*2.D0*X/RPAR1**2 
         UYYH=UH(1)*(4.D0*Y**2/RPAR2**4 - 2.D0/RPAR2**2)
         UYH=-UH(1)*2.D0*Y/RPAR2**2
C   Version 1:
         IF (IPAR.EQ.1) G(1)=-UXXH-UYYH+DEXP(U(1))-DEXP(DEXP(-Q))
C   Version 2:
         IF (IPAR.EQ.2) G(1)=-UXXH-UYYH-DEXP(U(1))+DEXP(DEXP(-Q))
C   Version 3:
         IF (IPAR.EQ.3) G(1)=-UXXH-UYYH
C
         DX(1)=1.D0
         DY(1)=1.D0
C
         CXY(1)=0.0D0
C
C
C 
      RETURN
      END
C
      SUBROUTINE DFUNS (NPDE,X,Y,U,UX,UY,RPAR1,RPAR2,IPAR,DX,DY,
     $                  DCDU,DCDXU,DCDYU,DGDU)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      PARAMETER(MNPDE=1)
C 
      DIMENSION U(NPDE),UX(NPDE),UY(NPDE)
      DIMENSION DCDU(MNPDE,MNPDE),DCDXU(MNPDE,MNPDE),DCDYU(MNPDE,MNPDE),
     $          DGDU(MNPDE,MNPDE)
      DIMENSION DX(NPDE),DY(NPDE)
      DIMENSION DUH(MNPDE)
C
         Q=X**2/RPAR1**2 + Y**2/RPAR2**2
         DUH(1)=0.1D0
         DUXXH=DUH(1)*(4.D0*X**2/RPAR1**4 - 2.D0/RPAR1**2)
         DUXH=-DUH(1)*2.D0*X/RPAR1**2 
         DUYYH=DUH(1)*(4.D0*Y**2/RPAR2**4 - 2.D0/RPAR2**2)
         DUYH=-DUH(1)*2.D0*Y/RPAR2**2
C   Version 1:
         IF (IPAR.EQ.1) DGDU(1,1)=-DUXXH-DUYYH+DEXP(U(1))
C   Version 2:
         IF (IPAR.EQ.2) DGDU(1,1)=-DUXXH-DUYYH-DEXP(U(1))
C   Version 3:
         IF (IPAR.EQ.3) DGDU(1,1)=-DUXXH-DUYYH
C
         DX(1) = 1.D0
         DY(1) = 1.D0
C
         DCDU (1,1) = 0.0D0
         DCDXU(1,1) = 0.0D0
         DCDYU(1,1) = 0.0D0
C
      RETURN
      END
C
      SUBROUTINE ITSOL( N, RHS, SOL, DEL, XW, MULJAC, PRECON,
     1       TOL, ITMAX, ITER, ERR, IERR, IOPT,
     2       LRWK, RWK, NRW, LIWK, IWK, NIW, LRWKU, RWKU, LIWKU, IWKU )
      INTEGER N
      DOUBLE PRECISION RHS(N), SOL(N), DEL(N), XW(N)
      EXTERNAL MULJAC, PRECON
      DOUBLE PRECISION TOL,ERR
      INTEGER ITMAX, ITER, IERR
      INTEGER IOPT(50)
      INTEGER LRWK
      DOUBLE PRECISION RWK(LRWK)
      INTEGER NRW, LIWK
      INTEGER IWK(LIWK) 
      INTEGER NIW, LRWKU
      DOUBLE PRECISION RWKU(LRWKU) 
      INTEGER LIWKU
      INTEGER IWKU(LIWKU)
C     ____________________________________________________________
C
C*    Parameters list description
C     ===========================
C
C         N           Int    The number of vector components
C         RHS(N)      Double The right hand side of the system (input)
C                            Must not be altered by ITSOL !
C       * SOL(N)      Double The array to get the approximate solution
C                            vector (output)
C                            On input: the startiterate
C       * DEL(N)      Double Must get on output the difference between
C                            the start- and the final iterate (in case
C                            of continuation iteration (IOPT(1).EQ.1):
C                            add the additional difference)
C                            On input supplied by GIANT with the zero
C                            vector, if a new iteration starts, and
C                            with the latest output from ITSOL, if an
C                            iteration will be continued.
C       * XW(N)       Double Array of scaling values for the solution 
C                            vector (may be altered)
C                            On input: The current scaling values of 
C                            Newton iterate.
C         MULJAC      Ext    Name of Jacobian*vector product subroutine
C         PRECON      Ext    Name of preconditioner subroutine
C         TOL         Double Prescribed tolerance for the convergence 
C                            criterium.
C         ITMAX       Int    Maximum number of iterations allowed 
C                            (input).
C         ITER        Int    Number of iterations done to get the
C                            solution (output).
C         ERR         Double Error estimate of error in final 
C                            approximate solution. (output)
C         IERR        Int    Return error flag. Zero indicates, that
C                            no error occured. (output)
C      *  IOPT(50)    Int    Linear solver options array as supplied by
C                            GIANT. (input)
C                            Information about the type of stopping
C                            criterion actually activated should be
C                            stored to position IOPT(50) (on output).
C                            See IOPT - description of ITSOL in GIANT
C                            for details on it.
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
C     ____________________________________________________________
C
      INTEGER LOWI, LOWR
      PARAMETER (LOWI=11,LOWR=11)
      INTEGER KMAXDF,KMAXMI
      PARAMETER ( KMAXDF=9, KMAXMI=5 )
      DOUBLE PRECISION RHOS
      DOUBLE PRECISION SMALL
      PARAMETER (SMALL=1.0D-10)
      INTEGER ICOUNT
      SAVE ICOUNT
C
      DOUBLE PRECISION RHSNRM, EPSIN, H1
      INTEGER ITOL
      EXTERNAL DGMRES
      IERR  = 0
      IFINI = IOPT(33)
      IOPT(50) = IOPT(33)
      IPUNIT=IOPT(14)
      IPRINT=IOPT(13)
      IHUNIT = 0
      IF (IPRINT.LT.0) IHUNIT=IPUNIT
      IF (IOPT(1).EQ.0) ICOUNT=0
      IRB=LOWR-1
C
        RHSNRM = 0.0D0
        DO 10 J=1,N
          RWK(IRB+J) = SOL(J)
          RHSNRM = RHSNRM+RHS(J)**2
C!!          RWK(IRB+N+J)=1.0D0/XW(J)
C!!          RWK(IRB+2*N+J)=1.0D0/DMAX1(DABS(RHS(J)),SMALL)
10      CONTINUE
        RHSNRM=DSQRT(RHSNRM)
        RHOS=RWK(1)
        EPSIN=TOL/RHOS
        MAXITI=ITMAX
C
5     CONTINUE
        L1=LOWR+N
        LRWKR=LRWK-N-IRB
C!!        L1=LOWR+3*N
C!!        LRWKR=LRWK-3*N-IRB
        IF (IOPT(31).EQ.0) IOPT(31)=KMAXDF
        KMAX = IOPT(31)
        IF (IOPT(31).LT.-1) KMAX = 0
        IF (IOPT(31).EQ.-1) THEN
          H1 = DBLE(FLOAT(N+3))
          KMAX = IDINT( DSQRT( H1*H1/4.0D0 + DBLE(FLOAT(LRWKR-6*N-1)))
     $                  - H1/2.0D0)
          KMAX = MAX0(KMAX,0)
          IF ( KMAX.LT.KMAXMI ) THEN
            IF (MPRERR.GE.1) WRITE (LUERR,10005) KMAX
10005       FORMAT(1X,'Workspace optimal KMAX would be ',I7,
     $                ' - but is too small')
            KMAX = KMAXMI
          ENDIF 
          IOPT(31) = KMAX
        ENDIF
        NRW = 1 + N*(KMAX+6) + KMAX*(KMAX+3)
        NIW = 20
        IF (IPRINT.LT.0) THEN
          WRITE(IPUNIT,1100) KMAX 
1100      FORMAT(6X,'KMax = ',I5)     
        ENDIF
        IF (NRW.GT.LRWKR) THEN
          IERR = -10
          IF (IPRINT.GT.0) WRITE(IPUNIT,10000) 'Real',NRW-LRWKR
10000     FORMAT(' ITSOL - ',A,' Workspace exhausted,',
     $           ' at least more required: ',I6)
        ENDIF
        IF (NIW.GT.LIWK) THEN
          IERR = -10
          IF (IPRINT.GT.0) WRITE(IPUNIT,10000) 'Integer',NIW-LIWK
        ENDIF
        IF (IERR.NE.0) RETURN
        IWK(1)=KMAX
        IWK(2)=IWK(1)
C       iwk(3):
C       0 : no scaling;  1 = column scaling; 2 = row scaling; 3 = row&column scal.
        IWK(3)=0
C!!        IWK(3)=3
C       iwk(4):
C       0: no precond.; >0: right precond.; <0: left precond.
C!        IWK(4)=1
        IWK(4)=-1
        IWK(5)=INT(MAXITI/KMAX)-1
C       IWK(5)=10
C       itol:
C       0: no/right precond. and first stop criterium; 1: (same as 0 ???)
C       2: left precond. and second stop criterium
C       3: no/left precond. and third stop test
        ITOL = 0
C
C       Extractions from DGMRES subroutine documentation :
C       . . .
C       . . .
C         If ITOL=0, then ERR = norm(SB*(B-A*X(L)))/norm(SB*B),
C                               for right or no preconditioning, and
C                         ERR = norm(SB*(M-inverse)*(B-A*X(L)))/
C                                norm(SB*(M-inverse)*B),
C                               for left preconditioning.
C         If ITOL=1, then ERR = norm(SB*(B-A*X(L)))/norm(SB*B),
C                               since right or no preconditioning
C                               being used.
C         If ITOL=2, then ERR = norm(SB*(M-inverse)*(B-A*X(L)))/
C                                norm(SB*(M-inverse)*B),
C                               since left preconditioning is being
C                               used.
C         If ITOL=3, then ERR =  Max  |(Minv*(B-A*X(L)))(i)/x(i)|
C                               i=1,n
C       . . .
C       . . .
C       DGMRES solves a linear system A*X = B rewritten in the form:
C
C        (SB*A*(M-inverse)*(SX-inverse))*(SX*M*X) = SB*B,
C
C       with right preconditioning, or
C
C        (SB*(M-inverse)*A*(SX-inverse))*(SX*X) = SB*(M-inverse)*B,
C
C       with left preconditioning, where A is an N-by-N double 
C       precision matrix,
C       X  and  B are N-vectors,   SB and SX   are  diagonal scaling
C       matrices,   and M is  a preconditioning    matrix.   . . .
C       . . .
C       . . .
C       . . .                                                    The
C       convergence criteria for stopping the  iteration is based on
C       the size  of the  scaled norm of  the residual  R(L)  =  B -
C       A*X(L).  The actual stopping test is either:
C
C               norm(SB*(B-A*X(L))) .le. TOL*norm(SB*B),
C
C       for right preconditioning, or
C
C               norm(SB*(M-inverse)*(B-A*X(L))) .le. 
C                       TOL*norm(SB*(M-inverse)*B),
C
C       for left preconditioning, where norm() denotes the euclidean
C       norm, and TOL is  a positive scalar less  than one  input by
C       the user.  . . .
C       . . .
        NELT = IWKU(6)
        NZMAX = IWKU(5)
        NUSRWK = IWKU(3)
        LOWIND = LOWI+IWKU(4)
C!!        CALL DGMRES(N, RHS, SOL, NELT, IWKU(LOWIND),
C!!     $                IWKU(LOWIND+NZMAX), RWKU(NUSRWK+1), 0, MULJAC,
C!!     $                 PRECON, ITOL, EPSIN, ITMAX, ITER, ERR, IERR, 
C!!     $                 IHUNIT, RWK(LOWR+2*N), RWK(LOWR+N), RWK(L1), 
C!!     $                 LRWKR, IWK, LIWK, RWKU, IWKU )
        CALL DGMRES(N, RHS, SOL, NELT, IWKU(LOWIND), IWKU(LOWIND+NZMAX), 
     $                RWKU(NUSRWK+1), 0, MULJAC, PRECON,
     $                ITOL, EPSIN, ITMAX, ITER, ERR, IERR, IHUNIT, 
     $                RROWSC, RCOLSC, RWK(L1), LRWKR, IWK, LIWK, 
     $                RWKU, IWKU )
        ICOUNT=ICOUNT+ITER
        ITER=ICOUNT
        IF (IERR.NE.0) THEN
C?           IF (IERR.GT.0) IERR=IERR+10
           IF (IERR.EQ.-1 .AND.IPRINT.GT.0) 
     $       WRITE(IPUNIT,*) ' GMRES - Insufficient real Ws, more ',
     $         ' required :',IWK(6)-LRWKR
          RETURN
        ENDIF
        IF (IPRINT.NE.0 .AND. IOPT(1).EQ.0)
     1    WRITE(IPUNIT,1090) ITER,ERR
1090    FORMAT(10X,'LinSol:   ','Iter:',I6,'  EstPrec:',D11.2)
        IF (IPRINT.NE.0 .AND. IOPT(1).EQ.1)
     1    WRITE(IPUNIT,1091) ITER,ERR
1091  FORMAT(10X,' Cont.:   ','Iter:',I6,'  EstPrec:',D11.2)
      IF (MAXITI.LE.0) IERR=2
      IF (IERR.EQ.4.AND.LTYP.EQ.2) IERR=0
      IF (ITER+1.LE.ITMAX .AND. IERR.EQ.0) THEN
        IERR = 0
      ELSE
        IF (IERR.EQ.0) IERR = -9
      ENDIF
      DO 90 J=1,N
        DEL(J) = DEL(J)+SOL(J)-RWK(IRB+J)
90    CONTINUE
      ERR = ERR*RHOS
C 
C     End of subroutine NJITSL
      RETURN
      END
C
      SUBROUTINE SLPREC(N, B, X, NELT, IA, JA, A, ISYM, RWK, IWK )
C*    begin prologue precon
      INTEGER N
      DOUBLE PRECISION B(N), X(N)
      INTEGER NELT
      INTEGER IA(NELT),JA(NELT)
      DOUBLE PRECISION A(NELT)
      INTEGER ISYM
      DOUBLE PRECISION RWK(*)
      INTEGER IWK(*)
C     ____________________________________________________________
C
C*    summary :
C
C     P R E C O N - call a preconditioner subroutine.
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
C     ____________________________________________________________
C*    end prologue
C
      EXTERNAL PRECON
C
      CALL PRECON( N, B, X, RWK, IWK )
C     end of subroutine SLPREC
      RETURN
      END
C
