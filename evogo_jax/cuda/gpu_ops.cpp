#include <pybind11/pybind11.h>
#include "DistanceKernel.h"
#include "helpers.h"

template <typename T> pybind11::bytes PackDescriptor(const T& descriptor) {
	return pybind11::bytes(PackDescriptorAsString(descriptor));
}

template <typename T> pybind11::capsule EncapsulateFunction(T* fn) {
	return pybind11::capsule(bit_cast<void*>(fn), "xla._CUSTOM_CALL_TARGET");
}

pybind11::dict CrossDistRegistrations() {
	pybind11::dict dict;
	dict["cross_dist_forward"] =
		EncapsulateFunction(crossDist_forward);
	dict["cross_dist_backward"] =
		EncapsulateFunction(crossDist_backward);
	return dict;
}

PYBIND11_MODULE(gpu_ops, m) {
	m.def("get_cross_dist_registrations", &CrossDistRegistrations);
	m.def("create_cdist_descriptor",
		[](double p, int batches, int dim, int lenA, int lenB, ElementType type)
		{
			return PackDescriptor(DistDescriptor(p, batches, dim, lenA, lenB, type));
		});
	m.def("get_temp_size", &getTempSize);

	pybind11::enum_<ElementType>(m, "ElementType")
		.value("F32", ElementType::F32)
		.value("F64", ElementType::F64);
}