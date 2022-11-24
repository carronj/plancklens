! jcarron March 2021
module gl
    implicit none
    double precision, parameter :: ATOL=1e-15, RTOL=1e-15, PI_4=0.78539816339744831d0, PI_SGL=3.141592654d0
    double precision, parameter :: stirling(1:10) = (/    1.0d0, 0.08333333333333333d0, 0.003472222222222222d0, &
                                    -0.0026813271604938273d0,   -0.00022947209362139917d0, 0.0007840392217200666d0,&
                                    6.972813758365857d-5, -0.0005921664373536939d0,  -5.171790908260592d-5,&
                                    0.0008394987206720873d0/) ! Stirlings expansion parameters
    contains

        subroutine startz(n, k, z) ! first guess at root number k
            implicit none
            integer, intent(in) :: n, k
            double precision, intent(out) :: z
            double precision cosk
            ! This works just fine for all.
             z = dcos(PI_SGL*(k-0.25d0)/(n+0.5d0)) * (1.d0 - (n-1) / (2.d0 * n) ** 3) !pi not double precision on purpose
        end subroutine startz

        double precision function glnorm(n)
            ! This is just Gamma(n + 1) / Gamma (n + 1 + 1/2) sqrt{4\pi} in double precision really
            ! Turned out accurate enough calc of this prefactor failed even with loggamma.
            implicit none
            integer, intent(in) :: n  ! intention is here always larger than 200
            double precision an, ni, n12i, s, s12, t
            integer i
            s = 1.d0
            s12 = 1.d0
            ni = 1.d0
            n12i = 1.d0
            do i = 2, 10
                ni = ni / n
                n12i = n12i / (n + 0.5d0)
                s = s + stirling(i) * ni
                s12 = s12 + stirling(i) * n12i
            end do
            t = 0.d0
            an = 1.d0
            do i = 1, 10, 2
                an = an  / (2 * n)
                t = t - an / i
                an = an / (2 * n)
                t = t + an / (i + 1)
            end do
            glnorm =  dsqrt(dexp(1.d0) / (n + 0.5d0)) * s / s12 * dexp((n + 0.5d0) * t) / dsqrt(PI_4)
        end function glnorm

        subroutine legphn(n, k, th, norm, m, pn, pp)
            !Superfast evaluation of Legendre polynomials for high (> 100) n
            implicit none
            double precision, intent(in) :: th, norm  ! not cos(th); norm is the precomputed amma(n + 1) / Gamma(n + 3/2) (tricky)
            integer, intent(in) :: n, k  ! k: root index (see startz above)
            integer, intent(out) :: m
            double precision, intent(out) :: pn, pp
            double precision cn, an, cth, sth, coth, dth, sth2,  coeff, coeff1, incrp, cosm, sinm, cosm1, sinm1

            cth = dcos(th)
            sth = dsin(th)
            coth = cth / sth
            sth2 = 2d0 * sth
            cn = norm ! same for all points
            cn = cn / dsqrt(sth2)

            ! we always expect the argument below to be k pi  + O(1/n^2) correction.
            ! The error in the sign term will give an error in the derivative -> weights
            dth = th - 4d0 * PI_4*(k-0.25d0)/(n+0.5d0)  ! this should always be very small (1/n^2) ?

            !cosm = dcos( (n + 0.5d0) * th - PI_4)    k pi - pi/2 + (n+1/2)dth
            !sinm = dsin( (n + 0.5d0) * th - PI_4)    k pi - pi/2 + (n+1/2)dth
            sinm = -dcos( (n + 0.5d0) * dth)  ! no large argument here but ~ 1/n,
            cosm =  dsin( (n + 0.5d0) * dth)  ! no large argument here but ~ 1/n
            if (BTEST(k, 0)) then ! sign if k odd
                sinm = -sinm
                cosm = -cosm
            end if
            coeff  = cn
            incrp = coeff * (n * sinm  +  0.5d0 * (coth * cosm + sinm))
            m = 0
            pn = 0.d0
            pp = 0.d0
            an = 1.d0
            do while ( ((coeff > (ATOL * 0.5)) .or. (abs(incrp/pp) > RTOL)) .and. (m < 1000 ))
                pn = pn + coeff * cosm
                pp = pp + incrp ! first derivative dPn / dtht
                m = m + 1
                an = an * (m - 0.5d0) ** 2 / m / (n + m + 0.5d0) / sth2
                cosm1 = cosm * sth + sinm * cth
                sinm1 = sinm * sth - cosm * cth
                cosm = cosm1  ! cos( n+1/2 th + m(th -pi/2) - pi/4)
                sinm = sinm1
                coeff = cn * an
                incrp = coeff * (n * sinm  +  (m + 0.5d0) * (coth * cosm + sinm)) ! there is a typo in the reference paper in the hn-1 -> hn equation
            end do
            pp = -pp
        end subroutine legphn

        subroutine legp(n, x, pn, pp) ! standard 3 terms rec for Pn and dPn / d cth
        implicit none
        double precision, intent(in) :: x  ! cos(tht)
        integer, intent(in) :: n
        double precision, intent(out) :: pn, pp
        double precision p2, p3
        integer j
        pn = 1.0d0
        p2 = 0.0d0
        do j=1,n
            p3 = p2
            p2 = pn
            pn = ((2*j-1)*x*p2-(j-1)*p3)/j
        end do
        pp = n*(x*pn-p2)/(x*x-1)
        end subroutine legp

        subroutine legpalln(n, nx, x, pn) ! standard 3 terms rec for Pn, returns Pn(x) for all x ann all mutlipoles below n
        implicit none
        double precision, intent(in) :: x(nx)  ! cos(tht)
        integer, intent(in) :: n, nx
        double precision, intent(out) :: pn(0:n, 1:nx)
        double precision p2(1:nx), p3(1:nx)
        integer j
        pn(0, :) = 1.0d0
        p2(:) = 0.0d0
        do j=1,n
            p3 = p2
            p2 = pn(j-1, :)
            pn(j, :) = ((2*j-1)*x*p2-(j-1)*p3)/j
        end do
        end subroutine legpalln

end module gl

subroutine get_xgwg(x1, x2, x, w, n)
    ! Gauss-Legendre quadrature points and weights.
    ! Algorithm scaling linearly with number of points, using super-fast high-n expansion of Legendre Pol.
    ! (adapted from Hale and Townsend 2013)
    ! I see ~10ms for N~10000
  use gl, only: legp, legphn, startz, glnorm
  implicit none
  integer, intent(in) :: n
  double precision, intent(in) :: x1, x2
  double precision, dimension(n), intent(out) :: x, w
  integer :: i, j, m, mi, it
  double precision :: p1, p2, p3, pp, xl, xm, z, z1, th, th1, norm
  double precision, parameter :: eps=1d-15
  m = (n+1)/2
  if (n < 200) then
      mi = m
  else
      mi = 50
  end if
  xm = 0.5d0*(x2+x1)
  xl = 0.5d0*(x2-x1)
  ! exterior region: (this should dominate the errors; did not bother to change that)
  do i=1, mi
    call startz(n, i, z)
    z1 = 0.d0
    do while(abs(z-z1) > eps)
        call legp(n, z, p1, pp)
        z1 = z
        z = z1 - p1/pp
    end do
    x(i) = xm - xl*z
    x(n+1-i) = xm + xl*z
    w(i) = (2*xl)/((1-z*z)*pp*pp)
    w(n+1-i) = w(i)
  end do
  ! Interior region: (following Hale and Townsend. Really fast.
  norm = glnorm(n)
  do i=mi + 1, m
    call startz(n, i, z)
    th = dacos(z)
    th1 = 0.d0
    do while(abs(th-th1) > eps)
        call legphn(n, i, th, norm, it, p1, pp)
        th1 = th
        th = th1 - p1/ pp
    end do
    z = cos(th)
    x(i) = xm - xl*z
    x(n+1-i) = xm + xl*z
    w(i) = (2*xl)/(pp*pp) ! now pp is d/dth, not d/dcth
    w(n+1-i) = w(i)
  end do
end subroutine get_xgwg


module gridutils
    implicit none
    double precision, parameter :: SYMTOL=1e-14
    contains
        logical function symgrid(x, nx)
            ! assuming here for most if not all of my applications is either not at all or fully symmetric
            implicit none
            integer, intent(in) :: nx
            double precision, intent(in) :: x(nx)
            integer ix
            ix = 0
            symgrid = nx > 1
            do while(symgrid .and. (ix < nx/2 + 1))
                ix = ix + 1
                symgrid = symgrid .and. ( abs((x(ix) + x(nx - (ix - 1)))) < SYMTOL)
            end do
        end function symgrid
end module gridutils

module jacobi
    implicit none
    contains
    subroutine rescal_jacobi(s1, s2, lmax, rl)
        ! coefficients turning Jacobi polynomials onto Wigner small d's
        ! ~0.1 ms at most for high lmax
        ! These coeffs are (for integer a, b)
        ! ( (k + (a + b)) ! * k ! / (k + a)! / (k + b)! )^1/2
        implicit none
        integer, intent(in) :: s1, s2
        integer, intent(in) :: lmax
        double precision, intent(out) :: rl(0:lmax - max(abs(s1), abs(s2)))
        integer a, b, lmin, k
        double precision dmab

        a = abs(s1 - s2)
        b = abs(s1 + s2)
        lmin = max(abs(s1), abs(s2))
        dmab = max(a, b)
        rl(0) = 1d0
        do k = 1, min(a, b)
            rl(0) = rl(0) * dsqrt(1.d0 + dmab / k)
        end do
        do k = 0, lmax - lmin - 1
            rl(k + 1) = rl(k) * dsqrt(  (k + 1.d0 + (a + b) ) / (k + 1.d0 + a) * (k + 1.d0) / (k + 1.d0 + b))
        end do
    end subroutine rescal_jacobi

    subroutine anbncn_jacobi(a, b, lmax, an, bn, cn)
        !Returns 3-terms recursion relation coefficients for Jacobi Polynomials
        ! ~0.1ms at most even for high lmax
        implicit none
        integer, intent(in) :: lmax
        double precision, intent(in) :: a, b
        double precision apb
        integer il
        double precision, intent(out):: an(0:lmax-1), bn(0:lmax-1), cn(0:lmax-1)
        apb = a + b
        cn(0) = 0.
        an(0) = (1 + apb) / 2 * (2 + apb) / (1 + apb)
        do il = 1, lmax - 1
            an(il) = (2 * il + 1 + apb) / (2 * (il + 1)) * (2 * il + 2 + apb) / (il + 1 + apb)
            cn(il) = (il + a) * (il + b) / (il + 1) * (2 * il + 2 + apb) / (il + 1 + apb) / (2 * il + apb)
        end do
        if (abs(a) == abs(b)) then
            bn = 0.d0
        else
            do il = 0, lmax - 1
              bn(il) = 0.5d0 * apb * (a - b) * (2 * il + 1 + apb) / (il + 1) / (2 * il + apb) / (il + 1 + apb)
            end do
        end if
    end subroutine anbncn_jacobi
end module jacobi

module poly
    implicit none
    contains

        subroutine rescal_coeff(lmax, an, bn, cn, rn)
        !Arranges the 3-terms recurence coefficients if the wanted polynmomials are rescaled versions of the original coeffs.
        !This assumes lmax > 0
        implicit none
        double precision an(0:lmax-1), bn(0:lmax-1), cn(0:lmax-1), rn(0:lmax)
        integer lmax, l
        double precision rab(0:lmax-1), rc(0:lmax-1)

        do l = 0, lmax - 1
            rab(l) = rn(l + 1) / rn(l)
        end do
        rc(0) = 0.
        do l = 1, lmax -1
            rc(l) = rn(l + 1) / rn(l - 1)
        end do
        an = an * rab
        bn = bn * rab
        cn = cn * rc
        end subroutine rescal_coeff

        subroutine pol2pos(xi, nx, lmax, x, an, bn, cn, cl, p0)
            ! Computes position space \sum_l cl Pl(x) without openMP threading
            ! Ortho polynomial eval up to lmax with 3 term recurrence relation
            ! P_{n+1}(x) = (a_n x + b_n) P_{n}(x) - c_n P_{n-1}(x) (Andrews, Askey and Roy)
            ! Normalization for ortho pol can also be gained from recurrence relation (see same ref)
            implicit none
            double precision, intent(in) :: x(nx)
            double precision, intent(in) :: an(0:lmax-1), bn(0:lmax-1), cn(0:lmax-1), cl(0:lmax)
            double precision, intent(out) :: xi(nx)
            double precision :: pl(nx), plp1(nx), plm1(nx), p0
            integer, intent(in) :: nx, lmax
            integer :: l

            if (lmax == 0) then
                xi = cl(0) * p0
                return
            end if
            plm1 = 0.
            pl = p0
            plp1 = (an(0) * x + bn(0)) * pl
            xi = cl(0) * pl + cl(1) * plp1
            do l = 1, lmax - 1
                plm1 = pl
                pl = plp1
                plp1 = (an(l) * x + bn(l)) * pl - cn(l) * plm1
                xi = xi + plp1 * cl(l + 1)
            end do
        end subroutine pol2pos

        subroutine pol2pos_omp(xi, nx, lmax, x, an, bn, cn, cl, p0)
            ! Computes position space \sum_l cl Pl(x) with openMP
            ! Ortho polynomial eval up to lmax with 3 term recurrence relation
            ! P_{n+1}(x) = (a_n x + b_n) P_{n}(x) - c_n P_{n-1}(x) (Andrews, Askey and Roy)
            ! Normalization for ortho pol can also be gained from recurrence relation (see same ref)
            implicit none
            double precision, intent(in) :: x(nx)
            double precision, intent(in) :: an(0:lmax-1), bn(0:lmax-1), cn(0:lmax-1), cl(0:lmax)
            double precision, intent(out) :: xi(nx)
            double precision :: pl, plp1, plm1, p0, txi
            integer, intent(in) :: nx, lmax
            integer :: l, ix

            if (lmax == 0) then
                xi = cl(0) * p0
                return
            end if
        !$OMP PARALLEL DO DEFAULT(NONE) PRIVATE(ix, txi, plm1, pl, plp1, l) SHARED(xi, x, lmax, an, bn, cn, cl, p0, nx)
            do ix = 1, nx
                plm1 = 0.
                pl = p0
                plp1 = (an(0) * x(ix) + bn(0)) * pl
                txi = cl(0) * pl + cl(1) * plp1
                do l = 1, lmax - 1
                    plm1 = pl
                    pl = plp1
                    plp1 = (an(l) * x(ix) + bn(l)) * pl - cn(l) * plm1
                    txi = txi + plp1 * cl(l + 1)
                end do
                xi(ix) = txi
            end do
        !$OMP END PARALLEL DO
        end subroutine pol2pos_omp

        subroutine pol2pos_omp_zsym(xi, nx, lmax, x, an, bn, cn, cl, p0)
            ! Computes position space \sum_l cl Pl(x) with openMP
            ! Ortho polynomial eval up to lmax with 3 term recurrence relation
            ! P_{n+1}(x) = (a_n x + b_n) P_{n}(x) - c_n P_{n-1}(x) (Andrews, Askey and Roy)
            ! Normalization for ortho pol can also be gained from recurrence relation (see same ref)
            ! This version fills half of the array assuming x(i) = -x(N-i) and Pl(1-x) = (-1)^l Pl(x) for a ~ factor of 2 speedup
            implicit none
            double precision, intent(in) :: x(nx)
            double precision, intent(in) :: an(0:lmax-1), bn(0:lmax-1), cn(0:lmax-1), cl(0:lmax)
            double precision, intent(out) :: xi(nx)
            double precision :: pl, plp1, plm1, p0, txi_p, txi_m
            integer, intent(in) :: nx, lmax
            integer :: l, ix

            if (lmax == 0) then
                xi = cl(0) * p0
                return
            end if
        !$OMP PARALLEL DO DEFAULT(NONE) PRIVATE(ix, txi_p, txi_m, plm1, pl, plp1, l) SHARED(xi, x, nx, lmax, an, bn, cn, cl, p0)
            do ix = 1, nx / 2 + MOD(nx, 2)
                plm1 = 0.
                pl = p0
                plp1 = (an(0) * x(ix) + bn(0)) * pl
                txi_p = cl(0) * pl   ! collects even multipoles only
                txi_m = cl(1) * plp1 ! collects odd mutipoles only
                do l = 1, lmax - 1
                    plm1 = pl
                    pl = plp1
                    plp1 = (an(l) * x(ix) + bn(l)) * pl - cn(l) * plm1
                    if ( BTEST(l, 0) ) then  ! same as ( mod(l, 2) > 0)
                        txi_p = txi_p + plp1 * cl(l + 1)
                    else
                        txi_m = txi_m + plp1 * cl(l + 1)
                    end if
                end do
                xi(ix) = txi_p + txi_m
                xi(nx - (ix - 1)) = txi_p - txi_m
            end do
        !$OMP END PARALLEL DO
        end subroutine pol2pos_omp_zsym

        subroutine pol2pos_clshw_omp(xi, nx, lmax, x, an, bn, cn, cl, p0)
            ! Computes position space \sum_l cl Pl(x)
            ! Ortho polynomial eval up to lmax with 3 term recurrence relation
            ! P_{n+1}(x) = (a_n x + b_n) P_{n}(x) - c_n P_{n-1}(x) (Andrews, Askey and Roy)
            ! Normalization for ortho pol can also be gained from recurrence relation (see same ref)
            ! This ues clenshaw recurrence
            ! Seems speedwise equivalent or sligthly worse than the forward scheme...
            implicit none
            double precision, intent(in) :: x(nx)
            double precision, intent(in) :: an(0:lmax-1), bn(0:lmax-1), cn(0:lmax-1), cl(0:lmax)
            double precision, intent(out) :: xi(nx)
            double precision :: yl, ylp1, ylp2, p0, tx
            integer, intent(in) :: nx, lmax
            integer :: l, ix
            if (lmax == 0) then
                xi = p0 * cl(0)
                return
            else if (lmax == 1) then
                xi = p0 * (cl(0) + cl(1) * (an(0) * x(ix) + bn(0)))
                return
            end if
        !$OMP PARALLEL DO DEFAULT(NONE) PRIVATE(ix, tx, yl, ylp1, ylp2, l) SHARED(xi, x, p0, nx, lmax, an, bn, cn, cl)
            do ix = 1, nx
                tx = x(ix)
                ylp2 = cl(lmax)  ! lmax
                ylp1 = (an(lmax - 1) * tx + bn(lmax - 1)) * ylp2 + cl(lmax - 1)  ! lmax - 1
                do l = lmax - 2, 1, -1
                    yl = (an(l) * tx + bn(l)) * ylp1 - cn(l+1) * ylp2 + cl(l)
                    ylp2 = ylp1
                    ylp1 = yl
                end do
                xi(ix) = p0 * (-cn(1) * ylp2 +  (an(0) * tx + bn(0)) * ylp1 + cl(0))
            end do
        !$OMP END PARALLEL DO
        end subroutine pol2pos_clshw_omp

        subroutine pos2pol(xi, nx, lmax, x, an, bn, cn, cl, p0, kmax)
            ! here lmax is not same as Jacobi's kmax, we fill cl up to lmax
            implicit none
            double precision, intent(in) :: x(nx)
            integer, intent(in) :: nx, lmax, kmax
            double precision, intent(in) :: an(0:kmax-1), bn(0:kmax-1), cn(0:kmax-1), xi(nx)
            double precision, intent(out) :: cl(0:lmax)
            double precision :: pl(nx), plp1(nx), plm1(nx), p0(nx)
            integer :: k, lmin
            if (kmax > lmax) then
                write(*, *) 'incompatible Wigner and Jacobi limits'
                error stop
            end if

            lmin = lmax - kmax
            cl(0:lmin-1) = 0.d0
            cl(lmin) = sum(xi * p0)
            if (lmax == lmin) then
                return
            end if
            pl = p0
            plp1 = (an(0) * x + bn(0)) * pl
            cl(lmin + 1) = sum(plp1 * xi)
            do k = 1, kmax - 1
                plm1 = pl
                pl = plp1
                plp1 = (an(k) * x + bn(k)) * pl - cn(k) * plm1
                cl(lmin + k + 1) = sum(plp1 * xi)
            end do
        end subroutine pos2pol

        subroutine pos2pol_omp(xi, nx, lmax, x, an, bn, cn, cl, p0, kmax)
            ! here lmax should not be confused with Jacobi's kmax, we fill cl up to lmax
            implicit none
            double precision, intent(in) :: x(nx)
            integer, intent(in) :: nx, lmax, kmax
            double precision, intent(in) :: an(0:kmax-1), bn(0:kmax-1), cn(0:kmax-1), xi(nx)
            double precision, intent(out) :: cl(0:lmax)
            double precision :: pl, plp1, plm1, p0(nx), cl_priv(0:kmax - 1)
            integer :: k, lmin, ix
            if (kmax > lmax) then
                write(*, *) 'incompatible Wigner and Jacobi limits'
                error stop
            end if

            cl = 0.d0
            lmin = lmax - kmax
            cl(lmin) = sum(xi * p0)
            if (lmax == lmin) then
                return
            end if
        !$OMP PARALLEL DEFAULT(NONE) PRIVATE(k, ix, plm1, pl, plp1, cl_priv) SHARED(lmin, x, xi, nx, kmax, p0, an, bn, cn, cl)
            cl_priv = 0.
        !$OMP DO
            do ix = 1, nx
                pl = p0(ix)
                plp1 = (an(0) * x(ix) + bn(0)) * pl
                cl_priv(0) = cl_priv(0) + plp1 * xi(ix)
                do k = 1, kmax - 1
                    plm1 = pl
                    pl = plp1
                    plp1 = (an(k) * x(ix) + bn(k)) * pl - cn(k) * plm1
                    cl_priv(k) = cl_priv(k) + plp1 * xi(ix)
                end do
            end do
        !$OMP END DO
        !$OMP CRITICAL
            cl(lmin +1:) = cl(lmin + 1:) + cl_priv
        !$OMP END CRITICAL
        !$OMP END PARALLEL

        end subroutine pos2pol_omp

        subroutine pos2pol_omp_zsym(xi, nx, lmax, x, an, bn, cn, cl, p0, kmax)
            ! here lmax should not be confused with Jacobi's kmax, we fill cl up to lmax
            ! This version fills the array assuming x(i) = -x(N-i) and Pl(1-x) = (-1)^l Pl(x) for a ~ factor of 2 speedup
            use gridutils, only : SYMTOL
            implicit none
            double precision, intent(in) :: x(nx)
            integer, intent(in) :: nx, lmax, kmax
            double precision, intent(in) :: an(0:kmax-1), bn(0:kmax-1), cn(0:kmax-1), xi(nx)
            double precision, intent(out) :: cl(0:lmax)
            double precision :: pl, plp1, plm1, p0(nx), clp1(0:kmax - 1), clp2(0:kmax - 1)
            integer :: k, lmin, ix
            if (kmax > lmax) then
                write(*, *) 'incompatible Wigner and Jacobi limits'
                error stop
            end if
            cl = 0.d0
            lmin = lmax - kmax
            cl(lmin) = sum(xi * p0)
            if (lmax == lmin) then
                return
            end if
        !$OMP PARALLEL DEFAULT(NONE) PRIVATE(k, ix, plm1, pl, plp1, clp1, clp2) SHARED(lmin, x, xi, nx, kmax, p0, an, bn, cn, cl)
            clp1 = 0.
            clp2 = 0.
        !$OMP DO
            do ix = 1, nx / 2
                pl = p0(ix)
                plp1 = (an(0) * x(ix) + bn(0)) * pl
                clp1(0) = clp1(0) + plp1 * xi(ix)
                clp2(0) = clp2(0) + plp1 * xi(nx - (ix-1))
                do k = 1, kmax - 1
                    plm1 = pl
                    pl = plp1
                    plp1 = (an(k) * x(ix) + bn(k)) * pl - cn(k) * plm1
                    clp1(k) = clp1(k) + plp1 * xi(ix)
                    clp2(k) = clp2(k) + plp1 * xi(nx - (ix - 1))
                end do
            end do
        !$OMP END DO
            do k = 0, kmax -1, 2 !parity flip for odd Jacobi multipoles (first entry corresponds to P_{n=1})
                clp2(k) = -clp2(k)
            end do
            clp1 = clp1 + clp2
        !$OMP CRITICAL
            cl(lmin +1:) = cl(lmin + 1:) + clp1
        !$OMP END CRITICAL
        !$OMP END PARALLEL
            ! adding the zero entry if present:
            if ( BTEST(nx, 0) ) then !same as mod(nx, 2) > 0
                ix = nx / 2 + 1
                if (abs(x(ix)) > SYMTOL) then
                    write(*, *) 'This should not have passed the symmetry scan'
                    error stop
                end if
                pl = p0(ix)  ! same as before with x = 0
                plp1 = bn(0) * pl
                cl(lmin + 1) = cl(lmin + 1)  + plp1 * xi(ix)
                do k = 1, kmax - 1
                    plm1 = pl
                    pl = plp1
                    plp1 = bn(k) * pl - cn(k) * plm1
                    cl(lmin + 1 + k) = cl(lmin + 1 + k) + plp1 * xi(ix)
                end do
            end if
        end subroutine pos2pol_omp_zsym

end module poly


subroutine wignerpos(xi, nx, lmax, cl, x, s1, s2)
    use gridutils, only: symgrid
    use jacobi, only:anbncn_jacobi, rescal_jacobi
    use poly, only: rescal_coeff, pol2pos, pol2pos_omp, pol2pos_omp_zsym
    implicit None
    double precision, intent(in) :: x(nx), cl(0:lmax)
    double precision, intent(out) :: xi(nx)
    integer, intent(in) :: s1, s2, nx, lmax
    double precision rn(0:lmax-max(abs(s1), abs(s2)))
    double precision an(0:lmax-max(abs(s1), abs(s2)) - 1)
    double precision bn(0:lmax-max(abs(s1), abs(s2)) - 1)
    double precision cn(0:lmax-max(abs(s1), abs(s2)) - 1)
    double precision a, b, p0, clm(0:lmax - max(abs(s1), abs(s2)))
    double precision, parameter :: PI4_i = 0.07957747154594767d0
    integer lmin, l, ix
    logical zsym

    lmin = max(abs(s1), abs(s2))
    if (lmin > lmax) then
        xi = 0.d0
        return
    end if
    !Jacobi Pol. parameters:
    a = abs(s1 - s2)
    b = abs(s1 + s2)
    ! Do I face a symmetric grid?
    ! one of the spin must be at least zero and we can use symmetry if the x grid is, with a plain factor of 2 gain
    zsym = (a == b) .and. (symgrid(x, nx))
    do l = lmin, lmax ! cl to pass against the Jacobi polynomials
        clm(l-lmin) = cl(l) * (2 * l + 1) * PI4_i !  cl (2l + 1) / 4pi
    end do
    p0 = 1.d0
    call anbncn_jacobi(a, b, lmax-lmin, an, bn, cn) ! Jacobi 3-terms recursion coefficients
    if (lmin > 0) then !at least one spin non-zero, we must rescale
        call rescal_jacobi(s1, s2, lmax, rn)
        if (lmax > lmin) then
            call rescal_coeff(lmax-lmin, an, bn, cn, rn)
        end if
        p0 = p0 * rn(0)
    end if
    if (zsym) then
        call pol2pos_omp_zsym(xi, nx, lmax-lmin, x, an, bn, cn, clm, p0)
    else
        if (nx > 1) then ! in principle, nx either 1 or very large here...
            call pol2pos_omp(xi, nx, lmax-lmin, x, an, bn, cn, clm, p0)
        else
            call pol2pos(xi, nx, lmax-lmin, x, an, bn, cn, clm, p0)
        end if
    end if
    if (a > 0) then
        xi = xi * (0.5 * (1 - x) ) ** (a * 0.5d0)  ! sin(x/2) ** a/2
    end if
    if (b > 0) then
        xi = xi * (0.5 * (1 + x) ) ** (b * 0.5d0)  ! cos(x/2) ** b/2
    end if
    if ( (s1 > s2) .AND. (BTEST(s1-s2, 0))) then
        xi = xi * (-1)
    end if
end subroutine wignerpos



subroutine wignercoeff(cl, xi, x, s1, s2, lmax, nx)
    ! This returns 2 pi \sum_x xi d^l_s1s2(x) for l up to lmax
    use gridutils, only : symgrid
    use jacobi, only:anbncn_jacobi, rescal_jacobi
    use poly, only: pos2pol_omp, pos2pol_omp_zsym, rescal_coeff
    implicit None
    double precision, intent(in) :: x(nx), xi(nx)
    double precision, intent(out) :: cl(0:lmax)
    integer, intent(in) :: s1, s2, nx, lmax
    double precision rn(0:lmax-max(abs(s1), abs(s2)))
    double precision an(0:lmax-max(abs(s1), abs(s2)) - 1)
    double precision bn(0:lmax-max(abs(s1), abs(s2)) - 1)
    double precision cn(0:lmax-max(abs(s1), abs(s2)) - 1)
    double precision p0(nx)
    double precision a, b
    double precision, parameter :: PI2 = 6.283185307179586d0
    integer lmin, l, ix
    logical zsym

    lmin = max(abs(s1), abs(s2))
    if (lmin > lmax) then
        cl = 0.d0
        return
    end if
    ! alpha and beta parameter of the Jacobi pol.
    a = abs(s1 - s2)
    b = abs(s1 + s2)
    ! Do I have a symmwetric xcolat grid?
    ! one of the spin at least is zero and we can use symmetry if the x grid is, with a plain factor of 2 gain
    zsym = (a == b) .and. (symgrid(x, nx))
    ! builds Jacobi 3-terms recursion coefficients:
    call anbncn_jacobi(a, b, lmax-lmin, an, bn, cn)
    ! Construction of the order zeroth Wigner polynomial in three steps:
    if ( (s1 > s2) .AND. (mod(s1-s2, 2) == 1) ) then
        p0 = -PI2
    else
        p0 = PI2
    end if
    if (a > 0) then
        p0 = p0 * (0.5 * (1 - x) ) ** (a * 0.5d0)  ! sin(x/2) ** a/2
    end if
    if (b > 0) then
        p0 = p0 * (0.5 * (1 + x) ) ** (b * 0.5d0)  ! cos(x/2) ** b/2
    end if
    if (lmin > 0) then !at least one spin non-zero. We need to rescal Jacobi polynomials
        call rescal_jacobi(s1, s2, lmax, rn)
        p0 = p0 * rn(0)
        if (lmax > lmin) then
            call rescal_coeff(lmax-lmin, an, bn, cn, rn)
        end if
    end if
     ! Supercrude split to decide which version of pos2pol to use.  In principle nx is always very large here...
    if (zsym) then
        call pos2pol_omp_zsym(xi, nx, lmax, x, an, bn, cn, cl, p0, lmax-lmin)
    else
        call pos2pol_omp(xi, nx, lmax, x, an, bn, cn, cl, p0, lmax-lmin)
    end if
end subroutine wignercoeff