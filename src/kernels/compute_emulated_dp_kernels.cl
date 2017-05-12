MSTRINGIFY(

// Stringifying requires a new line after hash defines

// Based on dsadd from DSFUN90, analysis by Norbert Juffa from NVIDIA.
// For a and b of opposite sign whose magnitude is within a factor of two
// of each other either variant below loses accuracy. Otherwise the result
// is within 1.5 ulps of the correctly rounded result with 48-bit mantissa.
float2 edp_add(float2 a, float2 b)
{
    float2 c;
    float t1, e, t2;

    //Compute dsa + dsb using Knuth's trick.
    t1 = a.x + b.x;
    e = t1 - a.x;
    t2 = ((b.x - e) + (a.x - (t1 - e))) + a.y + b.y;

    // The result is t1 + t2, after normalization.
    c.x = e = t1 + t2;
    c.y = t2 - (e - t1);

    return c;
}

// This function multiplies DS numbers A and B to yield the DS product C.
// Based on: Guillaume Da Graça, David Defour. Implementation of Float-Float
// Operators on Graphics Hardware. RNC'7, pp. 23-32, 2006.
float2 edp_mul(float2 a, float2 b)
{
    float2 c;
    float up, vp, u1, u2, v1, v2, mh, ml;
    uint tmp;

    //This splits a.x and b.x into high-order and low-order words.
    tmp = (*(uint *)&a) & ~0xFFF;	// Bit-style splitting from Reimar
    u1 = *(float *)&tmp;
    //up  = a.x * 4097.0f;
    //u1  = (a.x - up) + up;
    u2  = a.x - u1;
    tmp = (*(uint *)&b) & ~0xFFF;
    v1 = *(float *)&tmp;
    //vp  = b.x * 4097.0f;
    //v1  = (b.x - vp) + vp;
    v2  = b.x - v1;

    // Multilply a.x * b.x using Dekker's method.
    mh  = (a.x * b.x);
    ml  = ((((u1 * v1) - mh) + (u1 * v2)) + (u2 * v1)) + (u2 * v2);

    // Compute a.x * b.y + a.y * b.x
    ml  = ((a.x * b.y) + (a.y * b.x)) + ml;

    // The result is mh + ml, after normalization.
    c.x = up = mh + ml;
    c.y = (mh - up) + ml;
    return c;
}

float2 edp_mad(float2 a, float2 b, float2 c)
{
    return edp_add(edp_mul(a, b), c);
}

\n#undef MAD_4
\n#undef MAD_16
\n#undef MAD_64
\n
\n#define MAD_4(x, y)  x = edp_mad(y, x, y); y = edp_mad(x, y, x); x = edp_mad(y, x, y); y = edp_mad(x, y, x);
\n#define MAD_16(x, y) MAD_4(x, y);          MAD_4(x, y);          MAD_4(x, y);          MAD_4(x, y);
\n#define MAD_64(x, y) MAD_16(x, y);         MAD_16(x, y);         MAD_16(x, y);         MAD_16(x, y);
\n

__kernel void compute_edp(__global float2 *ptr, float2 _A)
{
    float2 x = _A;
    float2 y = {(float)get_local_id(0), 0.0f};

    MAD_64(x, y);   MAD_64(x, y);
    MAD_64(x, y);   MAD_64(x, y);
    MAD_64(x, y);   MAD_64(x, y);
    MAD_64(x, y);   MAD_64(x, y);
    MAD_64(x, y);   MAD_64(x, y);
    MAD_64(x, y);   MAD_64(x, y);
    MAD_64(x, y);   MAD_64(x, y);
    MAD_64(x, y);   MAD_64(x, y);

    MAD_64(x, y);   MAD_64(x, y);
    MAD_64(x, y);   MAD_64(x, y);
    MAD_64(x, y);   MAD_64(x, y);
    MAD_64(x, y);   MAD_64(x, y);
    MAD_64(x, y);   MAD_64(x, y);
    MAD_64(x, y);   MAD_64(x, y);
    MAD_64(x, y);   MAD_64(x, y);
    MAD_64(x, y);   MAD_64(x, y);

    ptr[get_global_id(0)] = y;
}

\n

)

