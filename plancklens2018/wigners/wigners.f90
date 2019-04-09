subroutine HELLOfunc
  INTEGER :: NTHREADS, TID, OMP_GET_NUM_THREADS, OMP_GET_THREAD_NUM

  ! Fork a team of threads giving them their own copies of variables
  !$omp PARALLEL PRIVATE(NTHREADS, TID)

  ! Obtain thread number
  TID = OMP_GET_THREAD_NUM()
  write(*,*) 'Hello World from thread = ', TID
  ! Only master thread does this
  IF (TID == 0) THEN
    NTHREADS = OMP_GET_NUM_THREADS()
    write(*,*) 'Number of threads = ', NTHREADS

  END IF
  ! All threads join master thread and disband
  !$omp END PARALLEL
end

double precision function lngamma(z)
!       Uses Lanczos-type approximation to ln(gamma) for z > 0.
!       Reference:
!            Lanczos, C. 'A precision approximation of the gamma function', J. SIAM Numer. Anal., B, 1, 86-96, 1964.
!       Accuracy: About 14 significant digits except for small regions in the vicinity of 1 and 2.
!
!       Programmer: Alan Miller
!                   1 Creswick Street, Brighton, Vic. 3187, Australia
!       Latest revision - 17 April 1988
    implicit none
    double precision a(9), z, lnsqrt2pi, tmp
    integer j
    data a/0.9999999999995183d0, 676.5203681218835d0,  -1259.139216722289d0, 771.3234287757674d0,&
              -176.6150291498386d0, 12.50734324009056d0, -0.1385710331296526d0, 0.9934937113930748d-05,&
              0.1659470187408462d-06/

    data lnsqrt2pi/0.9189385332046727d0/

    if (z <= 0.d0) return

    lngamma = 0.d0
    tmp = z + 7.d0
    do j = 9, 2, -1
        lngamma = lngamma + a(j)/tmp
        tmp = tmp - 1.d0
    end do
    lngamma = lngamma + a(1)
    lngamma = log(lngamma) + lnsqrt2pi - (z+6.5d0) + (z-0.5d0)*log(z+6.5d0)
end

subroutine get_xgwg(x1, x2, x, w, n)
    ! gauleg.f90     P145 Numerical Recipes in Fortran
    ! compute x(i) and w(i)  i=1,n  Gauss-Legendre ordinates and weights
    ! on interval -1.0 to 1.0 (length is 2.0)
    ! use ordinates and weights for Gauss Legendre integration
    ! After f2py implicit argument wrapping: x, w = get_xgwg(x1, x2, n)
  implicit none
  integer, intent(in) :: n
  double precision, intent(in) :: x1, x2
  double precision, dimension(n), intent(out) :: x, w
  integer :: i, j, m
  double precision :: p1, p2, p3, pp, xl, xm, z, z1
  double precision, parameter :: eps=3.d-14

  m = (n+1)/2
  xm = 0.5d0*(x2+x1)
  xl = 0.5d0*(x2-x1)
  do i=1,m
    z = cos(3.141592654d0*(i-0.25d0)/(n+0.5d0))
    z1 = 0.0
    do while(abs(z-z1) > eps)
      p1 = 1.0d0
      p2 = 0.0d0
      do j=1,n
        p3 = p2
        p2 = p1
        p1 = ((2.0d0*j-1.0d0)*z*p2-(j-1.0d0)*p3)/j
      end do
      pp = n*(z*p1-p2)/(z*z-1.0d0)
      z1 = z
      z = z1 - p1/pp
    end do
    x(i) = xm - xl*z
    x(n+1-i) = xm + xl*z
    w(i) = (2.0d0*xl)/((1.0d0-z*z)*pp*pp)
    w(n+1-i) = w(i)
  end do
end subroutine get_xgwg

subroutine get_legpn(p, x, np, nx)
    ! Legendre polynomials at x up to order np.
    implicit None
    integer, intent(in) :: np, nx
    double precision, intent(in) :: x(nx)
    double precision, intent(out) :: p(nx, np + 1)
    integer :: i, j
    ! After f2py implicit argument wrapping: p = wigners.get_legpn(x, np)
    p(:, 1) = 1.d0
    if (np == 0) return
    p(:, 2) = x
    if (np == 1) return
    do i = 1, np - 1
        p(:, i + 2) = 2.* x * p(:, i + 1) - p(:, i) - (x * p(:, i + 1)- p(:, i) )/(i + 1.)
    end do
end subroutine get_legpn

subroutine get_wignerd(d, x, lmax, mp, m, nx)
    implicit None
    integer, intent(in) :: mp, m, lmax, nx
    double precision, intent(in) :: x(nx)
    double precision, intent(out) ::  d(0:lmax, nx)
    integer :: a, b, k, lmin, n, ix
    double precision :: alfbet, a2, b2, n2_ab2, norm, a0, ak_km1, akm1_km2

    k = - max(abs(m), abs(mp))
    lmin = -k
    if (k == m) then
        a = mp - m
        sgn = (-1) ** (mp - m)
    else if (k == -m) then
        a = m - mp
        sgn = 1
    else if (k == mp) then
        a = m - mp
        sgn = 1
    else
        a = mp - m
        sgn = (-1) ** (mp - m)
    end if
    b = -2 * _k - a
    lmax = max(lmax, lmin)
    n = lmax + k

    alfbet= a + b
    a2 = a * a
    b2 = b * b

    a0 = exp(0.5d0 * (lngamma(2d0 * lmin + 1) - lngamma(a + 1d0) - lngamma(2 * lmin - a + 1d0)))
    ak_km1 = sqrt((1. + 2 * lmin) / (1. + a) / (1. + b))
    !    /* a1 / a0. (ak is coefficient relating Jacobi to Wigner) */

    d(lmin, :) = sgn * a0 * ((1. - x) * 0.5) **(0.5 * a) * ((1. + x) * 0.5) ** (b * 0.5)

end subroutine get_wignerd