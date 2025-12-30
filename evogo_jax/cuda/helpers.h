#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <chrono>


class Timer
{
private:
	// Type aliases to make accessing nested type easier
	using clock_type = std::chrono::steady_clock;
	using seconi_type = std::chrono::duration<double, std::ratio<1> >;

	std::chrono::time_point<clock_type> m_beg;

public:
	Timer() : m_beg{ clock_type::now() }
	{
	}

	void reset()
	{
		m_beg = clock_type::now();
	}

	double elapsed() const
	{
		return std::chrono::duration_cast<seconi_type>(clock_type::now() - m_beg).count();
	}
};


#define constexpr_decl(return_type, function_name, ...) constexpr return_type function_name##_constexpr(__VA_ARGS__) noexcept

constexpr_decl(int, sqrtint, int x)
{
	if (x < 0)
		return -1; // error code for negative input
	double curr = x, prev = 0;
	while (curr != prev)
	{
		prev = curr;
		curr = (curr + x / curr) / 2;
	}
	return static_cast<int>(curr);
}


// https://en.cppreference.com/w/cpp/numeric/bit_cast
template <class To, class From>
typename std::enable_if<sizeof(To) == sizeof(From) &&
	std::is_trivially_copyable<From>::value&&
	std::is_trivially_copyable<To>::value,
	To>::type
	bit_cast(const From& src) noexcept {
	static_assert(std::is_trivially_constructible<To>::value,
		"This implementation additionally requires destination type to "
		"be trivially constructible");

	To dst;
	memcpy(&dst, &src, sizeof(To));
	return dst;
}

template <typename T> std::string PackDescriptorAsString(const T& descriptor) {
	return std::string(bit_cast<const char*>(&descriptor), sizeof(T));
}

template <typename T>
const T* UnpackDescriptor(const char* opaque, std::size_t opaque_len) {
	if (opaque_len != sizeof(T)) {
		throw std::runtime_error("Invalid opaque object size");
	}
	return bit_cast<const T*>(opaque);
}