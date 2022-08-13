#include "main.h"
#include "extdll.h"

#include <cpuid.h>
void cpuid(int info[4], int InfoType) {
	__cpuid_count(InfoType, 0, info[0], info[1], info[2], info[3]);
}

bool cpuSSESuport = false;

// AMXX API
void OnPluginsLoaded()
{
}

void OnAmxxAttach()
{
	int cpuInfo[4];
	cpuid(cpuInfo, 1);
	cpuSSESuport = (cpuInfo[3] & ((int)1 << 26)) || false;

	MF_AddNatives(fastmath_natives);
}

void OnAmxxDetach()
{
}

void ErrorWindows(char *text)
{
}