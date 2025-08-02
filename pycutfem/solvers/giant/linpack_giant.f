      SUBROUTINE DGEFA(A,LDA,N,IPVT,INFO)
      INTEGER LDA,N,IPVT(*),INFO
      DOUBLE PRECISION A(LDA,*)
C 
C     dgefa factors a double precision matrix by gaussian elimination.
C 
C     dgefa is usually called by dgeco, but it can be called
C     directly with a saving in time if  rcond  is not needed.
C     (time for dgeco) = (1 + 9/n)*(time for dgefa) .
C 
C     on entry
C 
C        a       double precision(lda, n)
C                the matrix to be factored.
C 
C        lda     integer
C                the leading dimension of the array  a .
C 
C        n       integer
C                the order of the matrix  a .
C 
C     on return
C 
C        a       an upper triangular matrix and the multipliers
C                which were used to obtain it.
C                the factorization can be written  a = l*u  where
C                L  IS A PRODUCT OF PERMUTATION AND UNIT LOWER
C                triangular matrices and  u  is upper triangular.
C 
C        ipvt    integer(n)
C                an integer vector of pivot indices.
C 
C        info    integer
C                = 0  normal value.
C                = k  if  u(k,k) .eq. 0.0 .  this is not an error
C                     condition for this subroutine, but it does
C                     indicate that dgesl or dgedi will divide by zero
C                     if called.  use  rcond  in dgeco for a reliable
C                     indication of singularity.
C 
C     linpack. this version dated 08/14/78 .
C     cleve moler, university of new mexico, argonne national lab.
C 
C     subroutines and functions
C 
C     blas daxpy,dscal,idamax
C 
C     internal variables
C 
      DOUBLE PRECISION T
      INTEGER IDAMAX,J,K,KP1,L,NM1
C 
C 
C     gaussian elimination with partial pivoting
C 
      INFO = 0
      NM1 = N - 1
      IF (NM1 .LT. 1) GO TO 70
      DO 60 K = 1, NM1
         KP1 = K + 1
C 
C        find l = pivot index
C 
         L = IDAMAX(N-K+1,A(K,K),1) + K - 1
         IPVT(K) = L
C 
C        zero pivot implies this column already triangularized
C 
         IF (A(L,K) .EQ. 0.0D0) GO TO 40
C 
C           interchange if necessary
C 
            IF (L .EQ. K) GO TO 10
               T = A(L,K)
               A(L,K) = A(K,K)
               A(K,K) = T
   10       CONTINUE
C 
C           compute multipliers
C 
            T = -1.0D0/A(K,K)
            CALL DSCAL(N-K,T,A(K+1,K),1)
C 
C           row elimination with column indexing
C 
            DO 30 J = KP1, N
               T = A(L,J)
               IF (L .EQ. K) GO TO 20
                  A(L,J) = A(K,J)
                  A(K,J) = T
   20          CONTINUE
               CALL DAXPY(N-K,T,A(K+1,K),1,A(K+1,J),1)
   30       CONTINUE
         GO TO 50
   40    CONTINUE
            INFO = K
   50    CONTINUE
   60 CONTINUE
   70 CONTINUE
      IPVT(N) = N
      IF (A(N,N) .EQ. 0.0D0) INFO = N
      RETURN
      END
      SUBROUTINE DGESL(A,LDA,N,IPVT,B,JOB)
      INTEGER LDA,N,IPVT(*),JOB
      DOUBLE PRECISION A(LDA,*),B(*)
C 
C     dgesl solves the double precision system
C     a * x = b  or  trans(a) * x = b
C     using the factors computed by dgeco or dgefa.
C 
C     on entry
C 
C        a       double precision(lda, n)
C                the output from dgeco or dgefa.
C 
C        lda     integer
C                the leading dimension of the array  a .
C 
C        n       integer
C                the order of the matrix  a .
C 
C        ipvt    integer(n)
C                the pivot vector from dgeco or dgefa.
C 
C        b       double precision(n)
C                the right hand side vector.
C 
C        job     integer
C                = 0         to solve  a*x = b ,
C                = nonzero   to solve  trans(a)*x = b  where
C                            trans(a)  is the transpose.
C 
C     on return
C 
C        b       the solution vector  x .
C 
C     error condition
C 
C        a division by zero will occur if the input factor contains a
C        zero on the diagonal.  technically this indicates singularity
C        but it is often caused by improper arguments or improper
C        setting of lda .  it will not occur if the subroutines are
C        called correctly and if dgeco has set rcond .gt. 0.0
C        or dgefa has set info .eq. 0 .
C 
C     to compute  inverse(a) * c  where  c  is a matrix
C     with  p  columns
C           call dgeco(a,lda,n,ipvt,rcond,z)
C           if (rcond is too small) go to ...
C           do 10 j = 1, p
C              call dgesl(a,lda,n,ipvt,c(1,j),0)
C        10 continue
C 
C     linpack. this version dated 08/14/78 .
C     cleve moler, university of new mexico, argonne national lab.
C 
C     subroutines and functions
C 
C     blas daxpy,ddot
C 
C     internal variables
C 
      DOUBLE PRECISION DDOT,T
      INTEGER K,KB,L,NM1
C 
      NM1 = N - 1
      IF (JOB .NE. 0) GO TO 50
C 
C        job = 0 , solve  a * x = b
C        first solve  l*y = b
C 
         IF (NM1 .LT. 1) GO TO 30
         DO 20 K = 1, NM1
            L = IPVT(K)
            T = B(L)
            IF (L .EQ. K) GO TO 10
               B(L) = B(K)
               B(K) = T
   10       CONTINUE
            CALL DAXPY(N-K,T,A(K+1,K),1,B(K+1),1)
   20    CONTINUE
   30    CONTINUE
C 
C        now solve  u*x = y
C 
         DO 40 KB = 1, N
            K = N + 1 - KB
            B(K) = B(K)/A(K,K)
            T = -B(K)
            CALL DAXPY(K-1,T,A(1,K),1,B(1),1)
   40    CONTINUE
      GO TO 100
   50 CONTINUE
C 
C        job = nonzero, solve  trans(a) * x = b
C        first solve  trans(u)*y = b
C 
         DO 60 K = 1, N
            T = DDOT(K-1,A(1,K),1,B(1),1)
            B(K) = (B(K) - T)/A(K,K)
   60    CONTINUE
C 
C        now solve trans(l)*x = y
C 
         IF (NM1 .LT. 1) GO TO 90
         DO 80 KB = 1, NM1
            K = N - KB
            B(K) = B(K) + DDOT(N-K,A(K+1,K),1,B(K+1),1)
            L = IPVT(K)
            IF (L .EQ. K) GO TO 70
               T = B(L)
               B(L) = B(K)
               B(K) = T
   70       CONTINUE
   80    CONTINUE
   90    CONTINUE
  100 CONTINUE
      RETURN
      END
      DOUBLE PRECISION FUNCTION DDOT(N,DX,INCX,DY,INCY)
C
C     RETURNS THE DOT PRODUCT OF DOUBLE PRECISION DX AND DY.
C     DDOT = SUM FOR I = 0 TO N-1 OF  DX(LX+I*INCX) * DY(LY+I*INCY)
C     WHERE LX = 1 IF INCX .GE. 0, ELSE LX = (-INCX)*N, AND LY IS
C     DEFINED IN A SIMILAR WAY USING INCY.
C
      DOUBLE PRECISION DX(1),DY(1)
      DDOT = 0.D0
      IF(N.LE.0)RETURN
      IF(INCX.EQ.INCY) IF(INCX-1) 5,20,60
    5 CONTINUE
C
C         CODE FOR UNEQUAL OR NONPOSITIVE INCREMENTS.
C
      IX = 1
      IY = 1
      IF(INCX.LT.0)IX = (-N+1)*INCX + 1
      IF(INCY.LT.0)IY = (-N+1)*INCY + 1
      DO 10 I = 1,N
         DDOT = DDOT + DX(IX)*DY(IY)
        IX = IX + INCX
        IY = IY + INCY
   10 CONTINUE
      RETURN
C
C        CODE FOR BOTH INCREMENTS EQUAL TO 1.
C
C
C        CLEAN-UP LOOP SO REMAINING VECTOR LENGTH IS A MULTIPLE OF 5.
C
   20 M = MOD(N,5)
      IF( M .EQ. 0 ) GO TO 40
      DO 30 I = 1,M
         DDOT = DDOT + DX(I)*DY(I)
   30 CONTINUE
      IF( N .LT. 5 ) RETURN
   40 MP1 = M + 1
      DO 50 I = MP1,N,5
         DDOT = DDOT + DX(I)*DY(I) + DX(I+1)*DY(I+1) +
     1   DX(I + 2)*DY(I + 2) + DX(I + 3)*DY(I + 3) + DX(I + 4)*DY(I + 4)
   50 CONTINUE
      RETURN
C
C         CODE FOR POSITIVE EQUAL INCREMENTS .NE.1.
C
   60 CONTINUE
      NS = N*INCX
          DO 70 I=1,NS,INCX
          DDOT = DDOT + DX(I)*DY(I)
   70     CONTINUE
      RETURN
      END
      SUBROUTINE DAXPY(N,DA,DX,INCX,DY,INCY)
C
C     OVERWRITE DOUBLE PRECISION DY WITH DOUBLE PRECISION DA*DX + DY.
C     FOR I = 0 TO N-1, REPLACE  DY(LY+I*INCY) WITH DA*DX(LX+I*INCX) +
C       DY(LY+I*INCY), WHERE LX = 1 IF INCX .GE. 0, ELSE LX = (-INCX)*N,
C       AND LY IS DEFINED IN A SIMILAR WAY USING INCY.
C
      DOUBLE PRECISION DX(1),DY(1),DA
      IF(N.LE.0.OR.DA.EQ.0.D0) RETURN
      IF(INCX.EQ.INCY) IF(INCX-1) 5,20,60
    5 CONTINUE
C
C        CODE FOR NONEQUAL OR NONPOSITIVE INCREMENTS.
C
      IX = 1
      IY = 1
      IF(INCX.LT.0)IX = (-N+1)*INCX + 1
      IF(INCY.LT.0)IY = (-N+1)*INCY + 1
      DO 10 I = 1,N
        DY(IY) = DY(IY) + DA*DX(IX)
        IX = IX + INCX
        IY = IY + INCY
   10 CONTINUE
      RETURN
C
C        CODE FOR BOTH INCREMENTS EQUAL TO 1
C
C
C        CLEAN-UP LOOP SO REMAINING VECTOR LENGTH IS A MULTIPLE OF 4.
C
   20 M = MOD(N,4)
      IF( M .EQ. 0 ) GO TO 40
      DO 30 I = 1,M
        DY(I) = DY(I) + DA*DX(I)
   30 CONTINUE
      IF( N .LT. 4 ) RETURN
   40 MP1 = M + 1
      DO 50 I = MP1,N,4
        DY(I) = DY(I) + DA*DX(I)
        DY(I + 1) = DY(I + 1) + DA*DX(I + 1)
        DY(I + 2) = DY(I + 2) + DA*DX(I + 2)
        DY(I + 3) = DY(I + 3) + DA*DX(I + 3)
   50 CONTINUE
      RETURN
C
C        CODE FOR EQUAL, POSITIVE, NONUNIT INCREMENTS.
C
   60 CONTINUE
      NS = N*INCX
          DO 70 I=1,NS,INCX
          DY(I) = DA*DX(I) + DY(I)
   70     CONTINUE
      RETURN
      END
      SUBROUTINE DSCAL(N,DA,DX,INCX)
C
C     REPLACE DOUBLE PRECISION DX BY DOUBLE PRECISION DA*DX.
C     FOR I = 0 TO N-1, REPLACE DX(1+I*INCX) WITH  DA * DX(1+I*INCX)
C
      DOUBLE PRECISION DA,DX(1)
      IF(N.LE.0)RETURN
      IF(INCX.EQ.1)GOTO 20
C
C        CODE FOR INCREMENTS NOT EQUAL TO 1.
C
      NS = N*INCX
          DO 10 I = 1,NS,INCX
          DX(I) = DA*DX(I)
   10     CONTINUE
      RETURN
C
C        CODE FOR INCREMENTS EQUAL TO 1.
C
C
C        CLEAN-UP LOOP SO REMAINING VECTOR LENGTH IS A MULTIPLE OF 5.
C
   20 M = MOD(N,5)
      IF( M .EQ. 0 ) GO TO 40
      DO 30 I = 1,M
        DX(I) = DA*DX(I)
   30 CONTINUE
      IF( N .LT. 5 ) RETURN
   40 MP1 = M + 1
      DO 50 I = MP1,N,5
        DX(I) = DA*DX(I)
        DX(I + 1) = DA*DX(I + 1)
        DX(I + 2) = DA*DX(I + 2)
        DX(I + 3) = DA*DX(I + 3)
        DX(I + 4) = DA*DX(I + 4)
   50 CONTINUE
      RETURN
      END
      INTEGER FUNCTION IDAMAX(N,DX,INCX)
C
C     FIND SMALLEST INDEX OF MAXIMUM MAGNITUDE OF DOUBLE PRECISION DX.
C     IDAMAX =  FIRST I, I = 1 TO N, TO MINIMIZE  ABS(DX(1-INCX+I*INCX))
C
      DOUBLE PRECISION DX(1),DMAX,XMAG
      IDAMAX = 0
      IF(N.LE.0) RETURN
      IDAMAX = 1
      IF(N.LE.1)RETURN
      IF(INCX.EQ.1)GOTO 20
C
C        CODE FOR INCREMENTS NOT EQUAL TO 1.
C
      DMAX = DABS(DX(1))
      NS = N*INCX
      II = 1
          DO 10 I = 1,NS,INCX
          XMAG = DABS(DX(I))
          IF(XMAG.LE.DMAX) GO TO 5
          IDAMAX = II
          DMAX = XMAG
    5     II = II + 1
   10     CONTINUE
      RETURN
C
C        CODE FOR INCREMENTS EQUAL TO 1.
C
   20 DMAX = DABS(DX(1))
      DO 30 I = 2,N
          XMAG = DABS(DX(I))
          IF(XMAG.LE.DMAX) GO TO 30
          IDAMAX = I
          DMAX = XMAG
   30 CONTINUE
      RETURN
      END
