# define noise covariance operator
# here just starting with independent identically distributed

noise_var = 1e-4

def Cnoise_inv(f):
    # identity operator divided by variance
    return f/(noise_var)
