SRC := hipsolver_Dsyevj.cpp

EXE := hipsolver_Dsyevj

all: $(EXE)

INC      := -I/opt/rocm/include
LD_FLAGS := -L/opt/rocm/lib -lamdhip64 -lrocblas  -lhipsolver -D__HIP_PLATFORM_AMD__
#  -lsyevj
#  -lrocsolver


%: %.cpp
	g++ $< -o $@ $(INC) $(LD_FLAGS)


.PHONY: clean
clean:
	rm -rf $(EXE)



