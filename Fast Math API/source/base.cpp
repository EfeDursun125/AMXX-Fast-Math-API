#include "main.h"
#include "extdll.h"
#include <emmintrin.h>

// Fast Square Root
static cell AMX_NATIVE_CALL FastSquareRoot(AMX* amx, cell* params)
{
	float number = amx_ctof(params[1]);
#ifdef __SSE2__
	return amx_ftoc(_mm_cvtss_f32(_mm_rsqrt_ss(_mm_load_ss(&number))));
#else
	long i;
	float x2, y;
	const float threehalfs = 1.5f;
	x2 = number * 0.5f;
	y = number;
	i = *(long*)&y;
	i = 0x5f3759df - (i >> 1);
	y = *(float*)&i;
	y = y * (threehalfs - (x2 * y * y));
	return amx_ftoc(y);
#endif
}

// Square Root
static cell AMX_NATIVE_CALL SquareRoot(AMX* amx, cell* params)
{
	float number = amx_ctof(params[1]);
#ifdef __SSE2__
	return amx_ftoc(_mm_cvtss_f32(_mm_sqrt_ss(_mm_load_ss(&number))));
#else
	long i;
	float x2, y;
	const float threehalfs = 1.5f;
	x2 = number * 0.5f;
	y = number;
	i = *(long*)&y;
	i = 0x5f3759df - (i >> 1);
	y = *(float*)&i;
	y = y * (threehalfs - (x2 * y * y));
	return amx_ftoc(y * number);
#endif
}

// Clamp Float
static cell AMX_NATIVE_CALL ClampFloat(AMX* amx, cell* params)
{
	float a = amx_ctof(params[1]);
	float b = amx_ctof(params[2]);
	float c = amx_ctof(params[3]);

#ifdef __SSE2__
	return amx_ftoc(_mm_cvtss_f32(_mm_min_ss(_mm_max_ss(_mm_load_ss(&a), _mm_load_ss(&b)), _mm_load_ss(&c))));
#else
	return amx_ftoc(a > c ? c : (a < b ? b : a));
#endif
}

// Clamp
static cell AMX_NATIVE_CALL Clamp(AMX* amx, cell* params)
{
	return (params[1] > params[3] ? params[3] : (params[1] < params[2] ? params[2] : params[1]));
}

// Max Float
static cell AMX_NATIVE_CALL MaxFloat(AMX* amx, cell* params)
{
	float a = amx_ctof(params[1]);
	float b = amx_ctof(params[2]);

#ifdef __SSE2__
	return amx_ftoc(_mm_cvtss_f32(_mm_max_ss(_mm_load_ss(&a), _mm_load_ss(&b))));
#else
	if (a > b)
		return amx_ftoc(a);
	else if (b > a)
		return amx_ftoc(b);
	return amx_ftoc(b);
#endif
}

// Max
static cell AMX_NATIVE_CALL Max(AMX* amx, cell* params)
{
	if (params[1] > params[2])
		return params[1];
	else if (params[2] > params[1])
		return params[2];
	return params[2];
}

// Min Float
static cell AMX_NATIVE_CALL MinFloat(AMX* amx, cell* params)
{
	float a = amx_ctof(params[1]);
	float b = amx_ctof(params[2]);

#ifdef __SSE2__
		return amx_ftoc(_mm_cvtss_f32(_mm_min_ss(_mm_load_ss(&a), _mm_load_ss(&b))));
#else
	if (a < b)
		return amx_ftoc(a);
	else if (b < a)
		return amx_ftoc(b);
	return amx_ftoc(b);
#endif
}

// Min
static cell AMX_NATIVE_CALL Min(AMX* amx, cell* params)
{
	if (params[1] < params[2])
		return params[1];
	else if (params[2] < params[1])
		return params[1];
	return params[2];
}

// Multiply Float
static cell AMX_NATIVE_CALL MultiplyFloat(AMX* amx, cell* params)
{
	float num1 = amx_ctof(params[1]);
	float num2 = amx_ctof(params[2]);
#ifdef __SSE2__
	return amx_ftoc(_mm_cvtss_f32(_mm_mul_ss(_mm_load_ss(&num1), _mm_load_ss(&num2))));
#else
	return amx_ftoc(num1 * num2);
#endif
}

// Multiply Int
static cell AMX_NATIVE_CALL Multiply(AMX* amx, cell* params)
{
#ifdef __SSE2__
	return _mm_cvtsi128_si32(_mm_add_epi32(_mm_loadu_si32(&params[1]), _mm_loadu_si32(&params[2])));
#else
	return params[1] * params[2];
#endif
}

// Add Float
static cell AMX_NATIVE_CALL AddFloat(AMX* amx, cell* params)
{
	float num1 = amx_ctof(params[1]);
	float num2 = amx_ctof(params[2]);
#ifdef __SSE2__
	return _mm_cvtss_f32(_mm_add_ss(_mm_load_ss(&num1), _mm_load_ss(&num2)));
#else
	return amx_ftoc(num1 + num2);
#endif
}

// Add Int
static cell AMX_NATIVE_CALL Add(AMX* amx, cell* params)
{
#ifdef __SSE2__
	return _mm_cvtsi128_si32(_mm_add_epi32(_mm_loadu_si32(&params[1]), _mm_loadu_si32(&params[2])));
#else
	return params[1] + params[2];
#endif
}

// Subtract Float
static cell AMX_NATIVE_CALL SubtractFloat(AMX* amx, cell* params)
{
	float num1 = amx_ctof(params[1]);
	float num2 = amx_ctof(params[2]);
	return amx_ftoc(num1 - num2);
}

// Subtract
static cell AMX_NATIVE_CALL Subtract(AMX* amx, cell* params)
{
	return params[1] - params[2];
}

// Divide Float
static cell AMX_NATIVE_CALL DivideFloat(AMX* amx, cell* params)
{
	float num1 = amx_ctof(params[1]);
	float num2 = amx_ctof(params[2]);
#ifdef __SSE2__
	return _mm_cvtss_f32(_mm_div_ss(_mm_load_ss(&num1), _mm_load_ss(&num2)));
#else
	return amx_ftoc(num1 / num2);
#endif
}

// Divide
static cell AMX_NATIVE_CALL Divide(AMX* amx, cell* params)
{
	return params[1] / params[2];
}

AMX_NATIVE_INFO fastmath_natives[] =
{
	{ "c_fast_square_root", FastSquareRoot },
	{ "c_square_root", SquareRoot },
	{ "c_clamp", ClampFloat },
	{ "c_clamp_int", Clamp },
	{ "c_max", MaxFloat },
	{ "c_max_int", Max },
	{ "c_min", MinFloat },
	{ "c_min_int", Min },
	{ "c_multiply", MultiplyFloat },
	{ "c_multiply_int", Multiply },
	{ "c_add", AddFloat },
	{ "c_add_int", Add },
	{ "c_subtract", SubtractFloat },
	{ "c_subtract_int", Subtract },
	{ "c_divide", DivideFloat },
	{ "c_divide_int", Divide },
	{ NULL, NULL },
};