#include "main.h"
#include "extdll.h"

// AMXX API
void OnPluginsLoaded()
{
}

void OnAmxxAttach()
{
	MF_AddNatives(fastmath_natives);
}

void OnAmxxDetach()
{
}

void ErrorWindows(char *text)
{
}