#include <cmath>

namespace misc
{
    // This function converts a power value to decibels (dB).
    double db(double x)
    {
    return 10.0 * std::log10(x);
    }

}