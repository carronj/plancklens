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

subroutine get_wignerd(d, x, lmax, m1, m2, nx)
    ! Small-d Wigner matrices d^l_{mp, m}(x) for l from 0 to lmax.
    ! Uses their Jacobi orthogonal polynomials representation.
    implicit None
    integer, intent(in) :: m1, m2, lmax, nx
    double precision, intent(in) :: x(nx)
    double precision, intent(out) ::  d(nx, 0:lmax)
    double precision, external :: lngamma
    integer :: a, b, lmin, n, ix, in, sgn
    double precision :: alfbet, a2, b2, n2_ab2, norm, a0, ak_km1, akm1_km2

    lmin = max(abs(m1), abs(m2))
    if (lmin == -m2) then
        a = m1 - m2
        sgn = (-1) ** (m1 - m2)
    else if (lmin == m2) then
        a = m2 - m1
        sgn = 1
    else if (lmin == -m1) then
        a = m2 - m1
        sgn = 1
    else
        a = m1 - m2
        sgn = (-1) ** (m1 - m2)
    end if
    b = 2 * lmin - a
    n = max(lmax, lmin) - lmin

    alfbet= a + b
    a2 = a * a
    b2 = b * b

    a0 = exp(0.5d0 * (lngamma(2d0 * lmin + 1d0) - lngamma(a + 1d0) - lngamma(2 * lmin - a + 1d0)))
    ak_km1 = sqrt((1.d0 + 2 * lmin) / (1.d0 + a) / (1.d0 + b))
    !  a1 / a0. (ak is coefficient relating Jacobi to Wigner)
    d(:, 0:lmin) = 0d0
    d(:, lmin) = sgn * a0 * ((1.d0 - x) * 0.5d0) ** (0.5d0 * a) * ((1. + x) * 0.5d0) ** (b * 0.5d0)
    if (n > 0) then
        d(:, lmin + 1) = ak_km1 * d(:, lmin) * 0.5d0 * (2 * (a + 1) +  (alfbet + 2d0) * (x - 1d0))
    end if
    do in = 1, n -1
        akm1_km2 = ak_km1
        ak_km1 = sqrt((1d0 + lmin * 2d0 / (in + 1d0)) / (1d0 + a / (in + 1d0)) / (1d0 + b/(in + 1d0)))
        n2_ab2 = 2 * in + alfbet
        norm = 2 * (in + 1) * (in + 1 + alfbet) * n2_ab2
        d(:, lmin + in + 1) = (((n2_ab2 + 1.d0) * ((n2_ab2 + 2.d0) * n2_ab2 * x + a2 - b2)) * ak_km1 * d(:, lmin + in) &
        - 2 * ( in + a) * (in + b) * (n2_ab2 + 2d0) * akm1_km2 * ak_km1 * d(:, lmin + in - 1)) / norm
    end do
end subroutine get_wignerd

subroutine get_rspace(xi, cl, x, m1, m2, nx, lmax)
    ! Position-space representation
    ! sum_l Cl (2l + 1) / 4pi d^l_{m1,m2}(x)
    implicit None
    integer, intent(in) :: m1, m2, lmax, nx
    double precision, intent(in) :: x(nx), cl(0:lmax)
    double precision, intent(out) ::  xi(nx)
    double precision :: d0(nx), d1(nx), d2(nx)
    double precision, external :: lngamma
    integer :: a, b, lmin, n, ix, in, sgn
    double precision :: alfbet, a2, b2, n2_ab2, norm, a0, ak_km1, akm1_km2
    lmin = max(abs(m1), abs(m2))
    if (lmin == -m2) then
        a = m1 - m2
        sgn = (-1) ** (m1 - m2)
    else if (lmin == m2) then
        a = m2 - m1
        sgn = 1
    else if (lmin == -m1) then
        a = m2 - m1
        sgn = 1
    else
        a = m1 - m2
        sgn = (-1) ** (m1 - m2)
    end if
    b = 2 * lmin - a
    n = max(lmax, lmin) - lmin

    alfbet= a + b
    a2 = a * a
    b2 = b * b

    a0 = exp(0.5d0 * (lngamma(2d0 * lmin + 1d0) - lngamma(a + 1d0) - lngamma(2 * lmin - a + 1d0)))
    ak_km1 = sqrt((1.d0 + 2 * lmin) / (1.d0 + a) / (1.d0 + b))
    !  a1 / a0. (ak is coefficient relating Jacobi to Wigner)
    ! lmin
    d0 = sgn * a0 * ((1.d0 - x) * 0.5d0) ** (0.5d0 * a) * ((1. + x) * 0.5d0) ** (b * 0.5d0)
    xi = (2 * lmin + 1) * cl(lmin) * d0
    if (n > 0) then
        ! lmin + 1
        d1 = ak_km1 * d0 * 0.5d0 * (2 * (a + 1) +  (alfbet + 2d0) * (x - 1d0))
        xi = xi + (2 * lmin + 3) * cl(lmin + 1) * d1
    end if
    do in = 1, n -1
        akm1_km2 = ak_km1
        ak_km1 = sqrt((1d0 + lmin * 2d0 / (in + 1d0)) / (1d0 + a / (in + 1d0)) / (1d0 + b/(in + 1d0)))
        n2_ab2 = 2 * in + alfbet
        norm = 2 * (in + 1) * (in + 1 + alfbet) * n2_ab2
        !lmin + in + 1
        d2 = (((n2_ab2 + 1.d0) * ((n2_ab2 + 2.d0) * n2_ab2 * x + a2 - b2)) * ak_km1 * d1 &
        - 2 * ( in + a) * (in + b) * (n2_ab2 + 2d0) * akm1_km2 * ak_km1 * d0) / norm
        xi = xi + (2 * (lmin + in + 1) + 1) * cl(lmin + in + 1) * d2
        d0 = d1
        d1 = d2
    end do
    xi = xi * (0.25d0 / 3.14159265358979323846d0)
end subroutine get_rspace

subroutine wignercoeff(h, cl1, cl2, s1i, s1o, s2i, s2o, lmax1, lmax2, Lmax)
    implicit None
    integer, intent(in) :: s1i, s2i, s1o, s2o, lmax1, lmax2, Lmax
    double precision, intent(in) :: cl1(0:lmax1), cl2(0:lmax2)
    double precision, intent(out) :: h(0:Lmax)
    double precision :: xi1( (lmax1 + lmax2 + Lmax) / 2), xi2( (lmax1 + lmax2 + Lmax) / 2)

    !lmaxtot = lmax1 + lmax2 + lmax_out
    !N = (lmaxtot + 2 - lmaxtot%2) // 2

end subroutine wignercoeff
