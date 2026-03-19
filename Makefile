CXX = clang++
CXXFLAGS = -std=c++17 -O3 -Wall
LDFLAGS = -framework Foundation -framework Metal -framework CoreGraphics -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph

# Metal Shader Compiler
METAL = xcrun -sdk macosx metal
METALLIB = xcrun -sdk macosx metallib

# Python Configuration
PYTHON ?= python
EXT_SUFFIX = $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
PY_TARGET = mps_accel_core$(EXT_SUFFIX)
SHADERS = default.metallib

all: $(SHADERS) $(PY_TARGET)

$(PY_TARGET): src/bindings.mm src/flux_graph.mm
	$(CXX) $(CXXFLAGS) -shared -dynamiclib -undefined dynamic_lookup \
	$(shell $(PYTHON) -m pybind11 --includes) \
	-I$(shell $(PYTHON) -c "import torch; from torch.utils import cpp_extension; print(cpp_extension.include_paths()[0])") \
	-I$(shell $(PYTHON) -c "import torch; from torch.utils import cpp_extension; print(cpp_extension.include_paths()[1])") \
	$^ -o $@ $(LDFLAGS) \
	-L$(shell $(PYTHON) -c "import torch; from torch.utils import cpp_extension; print(cpp_extension.library_paths()[0])") \
	-Wl,-rpath,$(shell $(PYTHON) -c "import torch; from torch.utils import cpp_extension; print(cpp_extension.library_paths()[0])") \
	-ltorch -ltorch_cpu -ltorch_python -lc10

default.metallib: src/flux_kernels.metal
	$(METAL) -std=metal3.1 -c $< -o flux_kernels.air
	$(METALLIB) flux_kernels.air -o $@
	rm flux_kernels.air

clean:
	rm -f $(PY_TARGET) $(SHADERS)
