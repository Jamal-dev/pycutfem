      PROGRAM MAINV2
C
C     Example program for usage of GIANT with user supplied
C     matrix times vector and preconditioner subroutines.
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      PARAMETER (SIG=1.0D-5)
      PARAMETER (MN=961, MNPDE=1, LRESER=50, KMAX=9)
C     parameter (lrwk=(kmax+11)*mn+2*kmax+77, liwk=50)
      PARAMETER (LRWK=19315, LIWK=50)
C     parameter(mnx=31, mny=31)
C     parameter(liwku=21, lrwku=5*n+mnx+mny+3)
      PARAMETER(LIWKU=21, LRWKU=4870)
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
      EXTERNAL FCN,JAC,MULJAC,PRECON
      CHARACTER*100 FILNAM
      CHARACTER*16 CID
      DOUBLE PRECISION RPAR1, RPAR2
      INTEGER IPAR
C
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
      IWKU(10)=NX+NY+3
      IWKU(12)=NX
      IWKU(13)=NY
      NRWU = NX+NY+3
      IF (LRWKU.LT.NRWU) THEN
        WRITE(6,10002) NRWU
10002   FORMAT(' User function real workspace too small',/,
     $         ' at least required: ',I7)
        STOP
      ENDIF
      IWKU(21) = IPAR
      RWKU(NX+NY+1) = RPAR1
      RWKU(NX+NY+2) = RPAR2
      RWKU(NX+NY+3) = SIG
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
C     linear solver : Good Broyden
      IOPT(8)=1
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
C       Set print units LUERR, LUMON, LULIN (iwku(7) for SLAPInt)
      IOPT(12)=IUMON
      IOPT(14)=IUMON
      IOPT(18)=IUMON
C     Set It. lin. sol. parameters
      IOPT(41)=KMAX
      RWK(41)=4.0D0
      RWK(42)=4.0D0
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
      WRITE (6,10001) RWKU(NX+NY+3), IOPT(13), IOPT(17),
     $                IOPT(41), RWK(41), RWK(42),
     $                IOPT(31), RTOL, IWK(31)
10001 FORMAT(/, ' ============= Parameter settings ============= ',/,
     $       /,1X,'sig=',D8.1,
     $       /,1X,'mprmon=',I1,'  mprlin=',I4,
     $       /,1X,'kmax=',I3,'  rhoord=',D8.1,'  rhosim=',D8.1,
     $       /,1X,'nonlin=',I1,'  rtol=',D8.1,'  itmax=',I4,
     $       //,' ============================================== ',//)
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
        CALL GIANT(N,FCN,JAC,U,USCAL,RTOL,IOPT,IERR,
     $              LIWK,IWK,LRWK,RWK,
     $              LIWKU,IWKU,LRWKU,RWKU,MULJAC,PRECON,IDUMMY)
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
      SUBROUTINE  FCN(N,U,URHS,RWKU,IWKU,NFCN,IFAIL)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INTEGER N, IFAIL
      DOUBLE PRECISION U(N), URHS(N)
      DOUBLE PRECISION RWKU(*)
      INTEGER IWKU(*)
      NPDE = IWKU(11)
      NX   = IWKU(12)
      NY   = IWKU(13)
      IF (NFCN.EQ.0) THEN
        IWKU(16)=0
      ELSE
        IWKU(16)=IWKU(16)+1
      ENDIF
      T=0.0D0
      CALL FCN0 (N,NPDE,NX,NY,RWKU(1),RWKU(NX+1),RWKU(NX+NY+1),
     $           RWKU(NX+NY+2),IWKU(21),U,URHS)
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
      SUBROUTINE JAC(FCN,N,U,UWGT,F,RWKU,IWKU,NJAC,IFAIL)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      EXTERNAL FCN
      INTEGER N
      DOUBLE PRECISION U(N),UWGT(N),F(N)
      DOUBLE PRECISION RWKU(*)
      INTEGER IWKU(*)
      INTEGER NJAC,IFAIL 
      PARAMETER(MNPDE=1)
      DIMENSION DX(MNPDE),DY(MNPDE),DGDU(MNPDE,MNPDE),DUH(MNPDE)
C
      NPDE = IWKU(11)
      NX   = IWKU(12)
      NY   = IWKU(13)
      ILOW1= IWKU(10)
      ILOW2= ILOW1+N
      ILOW3= ILOW2+N
      RPAR1 = RWKU(NX+NY+1)
      RPAR2 = RWKU(NX+NY+2)
      IPAR = IWKU(21)
      DO 100 I=1,N
      RWKU(ILOW1+I)=U(I)
      RWKU(ILOW2+I)=F(I)
100   CONTINUE
C
C  compute diagonal elements of Jacobian (for use by preconditioner)
C 
C  interior nodes
C 
      DO 2000 IY=2,NY-1
       YL=RWKU(NX+IY-1)
       YM=RWKU(NX+IY)
       YR=RWKU(NX+IY+1)
       DYL=YM-YL
       DYR=YR-YM
       UYYFAC=-2.0D0/(DYL*DYR)
       DO 1000 IX=2,NX-1
        XL=RWKU(IX-1)
        XM=RWKU(IX)
        XR=RWKU(IX+1)
        DXL=XM-XL
        DXR=XR-XM
        UXXFAC=-2.0D0/(DXL*DXR)
C 
C  compute terms
C
         Q=XM**2/RPAR1**2 + YM**2/RPAR2**2
         DUH(1)=0.1D0
         DUXXH=DUH(1)*(4.D0*XM**2/RPAR1**4 - 2.D0/RPAR1**2)
         DUXH=-DUH(1)*2.D0*XM/RPAR1**2 
         DUYYH=DUH(1)*(4.D0*YM**2/RPAR2**4 - 2.D0/RPAR2**2)
         DUYH=-DUH(1)*2.D0*YM/RPAR2**2
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
C  set up Jacobian diagonal elements at actual node
C
        IH=ILOW3+((IY-1)*NX+IX-1)*NPDE
        DO 200 J=1,NPDE
            RWKU(IH+J) = DGDU(J,J) + DX(J)*UXXFAC+DY(J)*UYYFAC
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
        IXA = ILOW3+(IX-1)*NPDE
        DO 3300 J=1,NPDE
          RWKU(IXA+J) = 1.0D0
3300  CONTINUE
C
C  x=xmin
      DO 3100 IY=2,NY-1
        IYA = ILOW3+(IY-1)*NX*NPDE
        DO 3100 J=1,NPDE
          RWKU(IYA+J) = 1.0D0
3100  CONTINUE
C
C  x=xmax
      DO 3200 IY=2,NY-1
        IYA = ILOW3+(IY*NX-1)*NPDE
        DO 3200 J=1,NPDE
          RWKU(IYA+J) = 1.0D0
3200  CONTINUE
C
C  y=ymax
      DO 3400 IX=1,NX
        IXA = ILOW3+((IX-1)+(NY-1)*NX)*NPDE
        DO 3400 J=1,NPDE
          RWKU(IXA+J) = 1.0D0
3400  CONTINUE
      RETURN
      END
C
      SUBROUTINE PRECON(N,RHS,X,RWKU,IWKU)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
C
C  preconditioner diagonal scaling 
C
      DOUBLE PRECISION X(N),RHS(N)
      DOUBLE PRECISION RWKU(*)
      INTEGER IWKU(*)
C
      ILOW3 = IWKU(10)+2*N
      DO 100 I=1,N
      X(I)=RHS(I)/RWKU(ILOW3+I)
100   CONTINUE
C
      RETURN
      END
      SUBROUTINE MULJAC(N,RHS,Y,RWKU,IWKU)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
C
C     matrix times vector approximation subroutine
C
      DOUBLE PRECISION Y(N),RHS(N)
      DOUBLE PRECISION RWKU(*)
      INTEGER IWKU(*)
C
C
      ILOW=IWKU(10)
      SIG=RWKU(ILOW)
      IP2=ILOW+1+N
      IP4=IP2+2*N
      IP5=IP4+N
C
C  compute  y=jac(u)*rhs by (f(u+sig*rhs)-f(u))/sig 
C   (u is saved value of act. sol. in jac)
C   (f(u) is saved value of act. rhs. in jac)
C
C  compute perturbed argument for f-eval.
C
      DO 100 I=1,N
      RWKU(IP4-1+I)=RWKU(ILOW+I)+SIG*RHS(I)
100   CONTINUE
C
C  compute function 
C
      CALL FCN(N,RWKU(IP4),RWKU(IP5),RWKU,IWKU,NFCN,IFAIL)
C
C  compute product
      DO 200 I=1,N
      Y(I)=(RWKU(IP5-1+I)-RWKU(IP2-1+I))/SIG
200   CONTINUE
C
      RETURN
      END
