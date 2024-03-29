
HIPCC = /opt/rocm/bin/hipcc

LIB_SHARED := librocsolver.so.0
all: $(LIB_SHARED)

LIB_CPP_SRC := \
				common/src/common_host_helpers.cpp \
				library/src/common/buildinfo.cpp \
				library/src/common/rocsolver_logger.cpp \
				library/src/lapack/roclapack_syevj_heevj.cpp \
				library/src/lapack/roclapack_syevj_heevj_batched.cpp \
				library/src/lapack/roclapack_syevj_heevj_strided_batched.cpp \
				library/src/lapack/roclapack_syev_heev_batched.cpp \
				library/src/lapack/roclapack_syev_heev_strided_batched.cpp \
				library/src/lapack/roclapack_sytrd_hetrd_batched.cpp \
				library/src/lapack/roclapack_sytrd_hetrd_strided_batched.cpp \
				library/src/lapack/roclapack_sytd2_hetd2_batched.cpp \
				library/src/lapack/roclapack_sytd2_hetd2_strided_batched.cpp \
				library/src/auxiliary/rocauxiliary_lacgv.cpp \
library/src/auxiliary/rocauxiliary_larf.cpp \
library/src/auxiliary/rocauxiliary_larfb.cpp \
library/src/auxiliary/rocauxiliary_larfg.cpp \
library/src/auxiliary/rocauxiliary_larft.cpp \
library/src/auxiliary/rocauxiliary_latrd.cpp \
library/src/auxiliary/rocauxiliary_org2l_ung2l.cpp \
library/src/auxiliary/rocauxiliary_org2r_ung2r.cpp \
library/src/auxiliary/rocauxiliary_orgql_ungql.cpp \
library/src/auxiliary/rocauxiliary_orgqr_ungqr.cpp \
library/src/auxiliary/rocauxiliary_orgtr_ungtr.cpp \
library/src/auxiliary/rocauxiliary_steqr.cpp \
library/src/auxiliary/rocauxiliary_sterf.cpp


SOLVER_OBJ = $(LIB_CPP_SRC:.cpp=.o)





CPP_FLAGS := -fPIC -DROCM_USE_FLOAT16
INC := -I./common/include -I/opt/rocm/include -I/opt/rocm/include/rocblas -I./library/src/include -I./library/src
# -I/usr/include/fmt
LD_FLAGS := -L/opt/rocm/lib -lamdhip64 -lrocblas -lhipsolver -lrocsolver -D__HIP_PLATFORM_AMD__ -L/usr/lib/x86_64-linux-gnu -lfmt

#/opt/rocm/bin/hipcc -c -fPIC common/src/common_host_helpers.cpp -o h.o -I ./common/include -I /opt/rocm/include -D__HIP_PLATFORM_AMD__ -DROCM_USE_FLOAT16 -L /usr/lib/x86_64-linux-gnu/ -lfmt

%.o: %.cpp
	$(HIPCC) $(CPP_FLAGS)  $< -c -o $@ $(INC)

$(LIB_SHARED): $(SOLVER_OBJ) common/src/common_host_helpers.o	
	$(HIPCC) -shared $(CPP_FLAGS) $^ -o $@ $(LD_FLAGS)



# $(CU_SRC:.cu=.o)
.PHONY: clean
clean:
	rm -rf $(SOLVER_OBJ) $(LIB_SHARED)

#export LD_LIBRARY_PATH=/home/hipper/ex_hipsolver_rocsolver_suitesparse/local_d/rocsolver/lib:/home/hipper/ex_hipsolver_rocsolver_suitesparse/local_d/hipsolver/lib

#export LD_LIBRARY_PATH=/home/hipper/ex_hipsolver_rocsolver_suitesparse/ex/extract_hipsolver_syevj/roc_syevj/rocsolver:/home/hipper/ex_hipsolver_rocsolver_suitesparse/local_d/hipsolver/lib



