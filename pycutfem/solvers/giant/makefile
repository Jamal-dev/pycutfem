FFLAGS = -O
OBJS  =  giant.o linalg_giant.o linpack_giant.o zibconst.o zibsec.o zibmon.o

giant1: main_giant1.o $(OBJS) easypack_giant.o
	$(FC) $(DFLAGS) main_giant1.o easypack_giant.o $(OBJS) -o giant1

giant2: main_giant2.o $(OBJS)
	$(FC) $(DFLAGS) main_giant2.o $(OBJS) -o giant2

giant3: main_giant3.o $(OBJS) easypack_giant.o gmres_giant.o
	$(FC) $(DFLAGS) main_giant3.o easypack_giant.o gmres_giant.o $(OBJS) -o giant3

clean:
	rm -f *.o giant1 giant2 giant3
