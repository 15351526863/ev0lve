
#ifndef STDAFX_H
#define STDAFX_H

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX

#define WIN32_NO_STATUS
#include <windows.h>
#undef WIN32_NO_STATUS

#include <windowsx.h>
#include <intrin.h>
#include <ntstatus.h>
#include <winternl.h>

#include <algorithm>
#include <array>
#include <bitset>
#include <cctype>
#include <cstdint>
#include <deque>
#include <functional>
#include <iomanip>
#include <map>
#include <numeric>
#include <optional>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <macros.h>
#include <util/macros.h>

namespace sdk
{
#include <sdk/generated.h>
}

#endif // STDAFX_H
