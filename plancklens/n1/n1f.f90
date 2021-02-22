double precision function wf(k, l1x, l2x, l1y, l2y, l1abs, l2abs, cltt, clte, clee, lmaxtt, lmaxte, lmaxee)
    implicit None
    integer, intent(in) :: l1abs, l2abs, lmaxtt, lmaxte, lmaxee
    character(len=3), intent(in) :: k
    double precision, intent(in) :: l1x, l2x, l1y, l2y
    double precision, intent(in) :: cltt(lmaxtt), clte(lmaxte), clee(lmaxee)
    double precision cos2p, sin2p

    if (k == 'ptt') then
        wf =  (cltt(l1abs) * ((l1x + l2x) * l1x + (l1y + l2y) * l1y)  &
                + cltt(l2abs) * ((l1x + l2x) * l2x + (l1y + l2y) * l2y))

    else if (k == 'pte') then
        cos2p = 2d0 * (l1x * l2x + l1y * l2y) ** 2 / ( (l1x ** 2 + l1y ** 2) * (l2x ** 2 + l2y ** 2) ) - 1.
        wf = (clte(l1abs) * cos2p * ((l1x + l2x) * l1x + (l1y + l2y) * l1y)&
                +clte(l2abs) * ( (l1x + l2x) * l2x + (l1y + l2y) * l2y))

    else if (k == 'pet') then
        cos2p = 2d0 * (l2x * l1x + l2y * l1y) ** 2 / ( (l2x ** 2 + l2y ** 2) * (l1x ** 2 + l1y ** 2) ) - 1.
        wf = (clte(l2abs) * cos2p * ((l2x + l1x) * l2x + (l2y + l1y) * l2y)&
                + clte(l1abs) * ((l2x + l1x) * l1x + (l2y + l1y) * l1y))

    else if (k == 'ptb') then
        sin2p = 2d0 * (l1x * l2x + l1y * l2y) * (-l1y * l2x + l1x * l2y)/ ((l1x ** 2 + l1y ** 2) * (l2x ** 2 + l2y **2))
        wf = ((clte(l1abs) * ((l1x + l2x) * l1x + (l1y + l2y) * l1y)) * sin2p)

    else if (k == 'pbt') then
        sin2p = 2d0 * (l2x * l1x + l2y * l1y) * (-l2y * l1x + l2x * l1y)/ ((l2x ** 2 + l2y ** 2) * (l1x ** 2 + l1y **2))
        wf = ((clte(l2abs) * ((l2x + l1x) * l2x + (l2y + l1y) * l2y)) * sin2p)

    else if (k == 'pee') then
        cos2p = 2d0 * (l1x * l2x + l1y * l2y) ** 2 / ( (l1x ** 2 + l1y ** 2) * (l2x ** 2 + l2y ** 2) ) - 1.
        wf = (clee(l1abs) * ((l1x + l2x) * l1x + (l1y + l2y) * l1y) + &
                clee(l2abs) * ((l1x + l2x) * l2x + (l1y + l2y) * l2y)) * cos2p

    else if (k == 'peb') then
        !Note clBB absent here. If adding it, note Hu astro-ph/0111606 paper has the wrong sign in front of ClBB.
        sin2p = 2d0 * (l1x * l2x + l1y * l2y) * (-l1y * l2x + l1x * l2y)/ ((l1x ** 2 + l1y ** 2) * (l2x ** 2 + l2y **2))
        wf = (clee(l1abs) * ((l1x + l2x) * l1x + (l1y + l2y) * l1y)) * sin2p

    else if (k == 'pbe') then
        !Note clBB absent here. If adding it, note Hu astro-ph/0111606 paper has the wrong sign in front of ClBB.
        sin2p = 2d0 * (l2x * l1x + l2y * l1y) * (-l2y * l1x + l2x * l1y)/ ((l2x ** 2 + l2y ** 2) * (l1x ** 2 + l1y **2))
        wf = (clee(l2abs) * ((l2x + l1x) * l2x + (l2y + l1y) * l2y)) * sin2p

    else if ((k == 'pbb') .or. k =='xbb') then
        wf = 0d0

    else if (k == 'xtt') then
        wf =  (cltt(l1abs) * (-(l1y + l2y) * l1x + (l1x + l2x) * l1y) &
                + cltt(l2abs) * (-(l1y + l2y) * l2x + (l1x + l2x) * l2y))

    else if (k == 'xte') then
        cos2p = 2d0 * (l1x * l2x + l1y * l2y) ** 2 / ( (l1x ** 2 + l1y ** 2) * (l2x ** 2 + l2y ** 2) ) - 1.
        wf = (clte(l1abs) * cos2p * ( -(l1y + l2y) * l1x + (l1x + l2x) * l1y)&
                + clte(l2abs) * (-(l1y + l2y) * l2x + (l1x + l2x) * l2y))

    else if (k == 'xet') then
        cos2p = 2d0 * (l2x * l1x + l2y * l1y) ** 2 / ( (l2x ** 2 + l2y ** 2) * (l1x ** 2 + l1y ** 2) ) - 1.
        wf = (clte(l2abs) * cos2p * (-(l2y + l1y) * l2x + (l2x + l1x) * l2y) &
                + clte(l1abs) * (-(l2y + l1y) * l1x + (l2x + l1x) * l1y))

    else if (k == 'xtb') then
        sin2p = 2d0 * (l1x * l2x + l1y * l2y) * (-l1y * l2x + l1x * l2y)/ ((l1x ** 2 + l1y ** 2) * (l2x ** 2 + l2y **2))
        wf = (clte(l1abs) * (-(l1y + l2y) * l1x + (l1x + l2x) * l1y)) * sin2p

    else if (k == 'xbt') then
        sin2p = 2d0 * (l2x * l1x + l2y * l1y) * (-l2y * l1x + l2x * l1y)/ ((l2x ** 2 + l2y ** 2) * (l1x ** 2 + l1y **2))
        wf = (clte(l2abs) * (-(l2y + l1y) * l2x + (l2x + l1x) * l2y)) * sin2p

    else if (k == 'xee') then
        cos2p = 2d0 * (l1x * l2x + l1y * l2y) ** 2 / ( (l1x ** 2 + l1y ** 2) * (l2x ** 2 + l2y ** 2) ) - 1.
        wf = (clee(l1abs) * (-(l1y + l2y) * l1x + (l1x + l2x) * l1y) + clee(l2abs) * &
                (-(l1y + l2y) * l2x + (l1x + l2x) * l2y)) * cos2p

    else if (k == 'xeb') then
        !Note clBB absent here. If adding it, note Hu astro-ph/0111606 paper has the wrong sign in front of ClBB.
        sin2p = 2d0 * (l1x * l2x + l1y * l2y) * (-l1y * l2x + l1x * l2y)/ ((l1x ** 2 + l1y ** 2) * (l2x ** 2 + l2y **2))
        wf = (clee(l1abs) * (-(l1y + l2y) * l1x + (l1x + l2x) * l1y)) * sin2p

    else if (k == 'xbe') then
        !Note clBB absent here. If adding it, note Hu astro-ph/0111606 paper has the wrong sign in front of ClBB.
        sin2p = 2d0 * (l2x * l1x + l2y * l1y) * (-l2y * l1x + l2x * l1y)/ ((l2x ** 2 + l2y ** 2) * (l1x ** 2 + l1y **2))
        wf = (clee(l2abs) * (-(l2y + l1y) * l2x + (l2x + l1x) * l2y)) * sin2p

    else if (k == 'stt') then
        wf = 1d0

    else if (k == 'ftt') then
        wf = cltt(l1abs) + cltt(l2abs)

    else if ((k == 'ste') .or. (k == 'set') .or. (k == 'stb') .or. (k == 'stb') .or. (k == 'seb') .or. (k == 'sbe'))then
        wf = 0d0

    else
        wf = 0d0
        write(*,*) 'Failed to find QE weight function '//k
    end if
end

subroutine n1(ret, Ls, nL, cl_kI, kA, kB, kI, cltt, clte, clee, clttfid, cltefid, cleefid, &
                    ftlA, felA, fblA, ftlB, felB, fblB, lminA, lmaxA, lminB, lmaxB, lmaxI, &
                    lmaxtt, lmaxte, lmaxee, lmaxttfid, lmaxtefid, lmaxeefid, dL, lps, nlps)
    implicit None
    integer, intent(in) :: Ls(nL), lmaxA, lmaxB, lmaxI, lminA, lminB, dL, nL
    integer, intent(in) :: lmaxtt, lmaxte, lmaxee, lmaxttfid, lmaxtefid, lmaxeefid
    integer, intent(in) :: nlps, lps(0:nlps-1)
    double precision, intent(out) :: ret(nL)
    character(len=3), intent(in) :: kA, kB ! QE keys
    character(len=1), intent(in) :: kI     ! anisotropy source key
    double precision, intent(in) :: cltt(lmaxtt), clee(lmaxee), clte(lmaxte), cl_kI(lmaxI)
    double precision, intent(in) :: clttfid(lmaxttfid), cleefid(lmaxeefid), cltefid(lmaxtefid)
    double precision, intent(in) :: ftlA(lmaxA), felA(lmaxA), fblA(lmaxA)
    double precision, intent(in) :: ftlB(lmaxB), felB(lmaxB), fblB(lmaxB)
    double precision, external :: n1L
    integer iL

    !$OMP PARALLEL
    !$OMP DO SCHEDULE(DYNAMIC,1)
    do iL = 1, nL
        ret(iL) = n1L(Ls(iL), cl_kI, kA, kB, kI, cltt, clte, clee, clttfid, cltefid, cleefid, &
                    ftlA, felA, fblA, ftlB, felB, fblB, lminA, lmaxA, lminB, lmaxB, lmaxI, &
                    lmaxtt, lmaxte, lmaxee, lmaxttfid, lmaxtefid, lmaxeefid, dL, lps, nlps)
    end do
    !$OMP ENDDO
    !$OMP END PARALLEL
end

double precision function n1L(L, cl_kI, kA, kB, kI, cltt, clte, clee, clttfid, cltefid, cleefid, &
                    ftlA, felA, fblA, ftlB, felB, fblB, lminA, lmaxA, lminB, lmaxB, lmaxI, &
                    lmaxtt, lmaxte, lmaxee, lmaxttfid, lmaxtefid, lmaxeefid, dL, lps, nlps)
    implicit None
    integer, intent(in) :: L, lmaxA, lmaxB, lmaxI, lminA, lminB, dL
    integer, intent(in) :: lmaxtt, lmaxte, lmaxee, lmaxttfid, lmaxtefid, lmaxeefid
    integer, intent(in) :: nlps, lps(0:nlps-1)
    ! lps is the anisotropy source multipole discretization.
    ! Planck 2018 used lps=(/1,2,12,22,32,42,52,62,72,82,92,102,132,162,192,222,252,282,312,342,372, &
    !    !                   402,432,462,492,522,552,652,752,852,952,1052,1152,1452,1752,2052,2352,2500/)
    character(len=3), intent(in) :: kA, kB ! QE keys
    character(len=1), intent(in) :: kI     ! anisotropy source key
    double precision, intent(in) :: cltt(lmaxtt), clee(lmaxee), clte(lmaxte), cl_kI(lmaxI)
    double precision, intent(in) :: clttfid(lmaxttfid), cleefid(lmaxeefid), cltefid(lmaxtefid)
    double precision, intent(in) :: ftlA(lmaxA), felA(lmaxA), fblA(lmaxA)
    double precision, intent(in) :: ftlB(lmaxB), felB(lmaxB), fblB(lmaxB)
    double precision, external :: wf ! QE weight functions


    double precision :: fal1(lmaxA), fal2(lmaxA), fal3(lmaxB), fal4(lmaxB)
    double precision ::  L1, L2, L3, L4, L1x, L1y, L2x, L2y, Lx, Ly, M_PI, L3x, L3y, L4x, L4y
    integer :: L1i, L2i, L3i, L4i, i
    double precision :: phi, dphi, dPh, PhiL_phi_dphi, fac, PhiLx, PhiLy, PhiL_phi, term1, term2
    integer :: nphi, phiIx, PhiLix, PhiLi, PhiL_nphi, PhiL_nphi_ix
    character(len=3) :: k13, k24, k14, k23
    double precision :: dlps(0:nlps-1)

    M_PI = 3.14159265358979323846d0
    Ly = 0d0
    Lx = float(L)

    k13 = kI // kA(2:2) // kB(2:2)
    k24 = kI // kA(3:3) // kB(3:3)
    k14 = kI // kA(2:2) // kB(3:3)
    k23 = kI // kA(3:3) // kB(2:2)

    dlps(0) = float(lps(1) - lps(0))
    do i = 1, nlps - 2
        dlps(i) = 0.5d0 * (lps(i + 1) - lps(i - 1))
    end do
    dlps(nlps - 1) = float(lps(nlps - 1) - lps(nlps - 2))

    if (kA(2:2) == 't') then
        fal1 = ftlA
    else if (kA(2:2) == 'e') then
        fal1 = felA
    else if (kA(2:2) == 'b') then
        fal1 = fblA
    else
        write(*,*) 'Failed to find cmb filter '//kA
    end if
    if (kB(2:2) == 't') then
        fal3 = ftlB
    else if (kB(2:2) == 'e') then
        fal3 = felB
    else if (kB(2:2) == 'b') then
        fal3 = fblB
    else
        write(*,*) 'Failed to find cmb filter '//kB
    end if
        if (kA(3:3) == 't') then
        fal2 = ftlA
    else if (kA(3:3) == 'e') then
        fal2 = felA
    else if (kA(3:3) == 'b') then
        fal2 = fblA
    else
        write(*,*) 'Failed to find cmb filter '//kA
    end if
    if (kB(3:3) == 't') then
        fal4= ftlB
    else if (kB(3:3) == 'e') then
        fal4 = felB
    else if (kB(3:3) == 'b') then
        fal4 = fblB
    else
        write(*,*) 'Failed to find cmb filter '//kB
    end if

    n1L = 0d0
    do L1i =  max(lminA, dL / 2), lmaxA, dL
        L1 = float(L1i)
        nphi = 2 * L1i + 1
        if (L1i > 3 * dL) then
            nphi = 2 * nint(0.5d0 * L1i  / float(dL)) + 1
        end if
        dphi = 2d0 * M_PI / nphi
        do phiIx = 0, (nphi - 1)/ 2
            phi = dphi * phiIx
            L1x = L1 * cos(phi)
            L1y = L1 * sin(phi)
            L2x = Lx - L1x
            L2y = Ly - L1y
            L2 = sqrt(L2x * L2x + L2y * L2y)
            if ( (L2 >= lminA) .AND. (L2 <= lmaxA) ) then
                L2i = nint(L2)
                !integral over (Lphi,Lphi_angle) according to lps grid.
                do PhiLix = 0, nlps - 1
                    PhiLi = lps(PhiLix)
                    dPh = dlps(PhiLix)
                    PhiL_nphi = 2 * PhiLi + 1
                    if (PhiLi > 20) then
                        PhiL_nphi = 2 * nint(0.5d0 * PhiL_nphi/dPh) + 1
                    end if
                    PhiL_phi_dphi = 2.d0 * M_PI / PhiL_nphi
                    fac  = (PhiL_phi_dphi * PhiLi * dPh) * (dphi * L1 * dL) / ( (2. * M_PI) ** 4. ) * 0.25d0
                    if (phiIx /= 0) then
                        fac = fac * 2d0
                    end if
                    fac = fac * wf(kA, L1x, L2x, L1y, L2y, L1i, L2i, clttfid, cltefid, cleefid, lmaxttfid, lmaxtefid, lmaxeefid)
                    fac = fac * fal1(L1i) * fal2(L2i)
                    do PhiL_nphi_ix = -(PhiL_nphi-1)/2, (PhiL_nphi-1)/2
                        PhiL_phi = PhiL_phi_dphi * PhiL_nphi_ix
                        PhiLx = PhiLi * cos(PhiL_phi)
                        PhiLy = PhiLi * sin(PhiL_phi)
                        L3x = PhiLx - L1x
                        L3y = PhiLy - L1y
                        L3 = sqrt(L3x*L3x + L3y*L3y)
                        if ((L3 >= lminB) .AND. (L3 <= lmaxB)) then
                            L3i = nint(L3)
                            L4x = -Lx - L3x
                            L4y = -Ly - L3y
                            L4 = sqrt(L4x * L4x + L4y * L4y)
                             if ((L4 >= lminB) .AND. (L4 <= lmaxB)) then
                                 L4i = nint(L4)
                                 ! wf(kA) and first two filters already absorded into 'fac'
                                 term1 = wf(kB, L3x, L4x, L3y, L4y, L3i, L4i, clttfid, cltefid, cleefid, &
                                                 lmaxttfid, lmaxtefid, lmaxeefid)&
                                         * wf(k13, L1x, L3x, L1y, L3y, L1i, L3i, cltt, clte, clee, &
                                                 lmaxtt, lmaxte, lmaxee)&
                                         * wf(k24, L2x, L4x, L2y, L4y, L2i, L4i, cltt, clte, clee, &
                                                 lmaxtt, lmaxte, lmaxee)&
                                         * fal3(L3i) * fal4(L4i)
                                 term2 = wf(kB, L4x, L3x, L4y, L3y, L4i, L3i, clttfid, cltefid, cleefid, &
                                           lmaxttfid, lmaxtefid, lmaxeefid)&
                                         * wf(k14, L1x, L3x, L1y, L3y, L1i, L3i, cltt, clte, clee, &
                                                 lmaxtt, lmaxte, lmaxee)&
                                         * wf(k23, L2x, L4x, L2y, L4y, L2i, L4i, cltt, clte, clee, &
                                                 lmaxtt, lmaxte, lmaxee)&
                                         * fal3(L4i) * fal4(L3i)
                                 n1L = n1L  + (term1 + term2) * fac * cl_kI(PhiLi)
                            end if
                        end if
                    end do
                end do
            end if
        end do
    end do
end


double precision function n1L_jtp(L, cl_kI, kA, kB, Xp, Yp, Ip, Jp, kI, cltt, clte, clee, clttfid, cltefid, cleefid, &
                    fXXp, fYYp, fIIp, fJJp, lminA, lmaxA, lminB, lmaxB, lmaxI, &
                    lmaxtt, lmaxte, lmaxee, lmaxttfid, lmaxtefid, lmaxeefid, dL, lps, nlps)
    implicit None
    integer, intent(in) :: L, lmaxA, lmaxB, lmaxI, lminA, lminB, dL
    integer, intent(in) :: lmaxtt, lmaxte, lmaxee, lmaxttfid, lmaxtefid, lmaxeefid
    integer, intent(in) :: nlps, lps(0:nlps-1)
    ! lps is the anisotropy source multipole discretization.
    ! Planck 2018 used lps=(/1,2,12,22,32,42,52,62,72,82,92,102,132,162,192,222,252,282,312,342,372, &
    !    !                   402,432,462,492,522,552,652,752,852,952,1052,1152,1452,1752,2052,2352,2500/)
    character(len=3), intent(in) :: kA, kB ! QE keys (XY IJ), e.g. 'ptt' for lensing gradient TT estimator
    character(len=1), intent(in) :: kI, Xp, Yp, Ip, Jp     ! anisotropy source key (typically 'p' for lensing gradient, 'x' for curl) and response fields
    double precision, intent(in) :: cltt(lmaxtt), clee(lmaxee), clte(lmaxte), cl_kI(lmaxI)
    double precision, intent(in) :: clttfid(lmaxttfid), cleefid(lmaxeefid), cltefid(lmaxtefid)
    double precision, intent(in) :: fXXp(lmaxA), fYYp(lmaxA), fIIp(lmaxB), fJJp(lmaxB)

    double precision, external :: wf ! QE weight functions


    double precision :: fal1(lmaxA), fal2(lmaxA), fal3(lmaxB), fal4(lmaxB)
    double precision ::  L1, L2, L3, L4, L1x, L1y, L2x, L2y, Lx, Ly, M_PI, L3x, L3y, L4x, L4y
    integer :: L1i, L2i, L3i, L4i, i
    double precision :: phi, dphi, dPh, PhiL_phi_dphi, fac, PhiLx, PhiLy, PhiL_phi, term1, term2
    integer :: nphi, phiIx, PhiLix, PhiLi, PhiL_nphi, PhiL_nphi_ix
    character(len=3) :: k13, k24, k14, k23
    double precision :: dlps(0:nlps-1)

    M_PI = 3.14159265358979323846d0
    Ly = 0d0
    Lx = float(L)

    k13 = kI // Xp // Ip
    k24 = kI // Yp // Jp
    k14 = kI // Xp // Jp
    k23 = kI // Yp // Ip

    dlps(0) = float(lps(1) - lps(0))
    do i = 1, nlps - 2
        dlps(i) = 0.5d0 * (lps(i + 1) - lps(i - 1))
    end do
    dlps(nlps - 1) = float(lps(nlps - 1) - lps(nlps - 2))

    fal1 = fXXp
    fal2 = fYYp
    fal3 = fIIp
    fal4 = fJJp

    n1L_jtp = 0d0
    do L1i =  max(lminA, dL / 2), lmaxA, dL
        L1 = float(L1i)
        nphi = 2 * L1i + 1
        if (L1i > 3 * dL) then
            nphi = 2 * nint(0.5d0 * L1i  / float(dL)) + 1
        end if
        dphi = 2d0 * M_PI / nphi
        do phiIx = 0, (nphi - 1)/ 2
            phi = dphi * phiIx
            L1x = L1 * cos(phi)
            L1y = L1 * sin(phi)
            L2x = Lx - L1x
            L2y = Ly - L1y
            L2 = sqrt(L2x * L2x + L2y * L2y)
            if ( (L2 >= lminA) .AND. (L2 <= lmaxA) ) then
                L2i = nint(L2)
                !integral over (Lphi,Lphi_angle) according to lps grid.
                do PhiLix = 0, nlps - 1
                    PhiLi = lps(PhiLix)
                    dPh = dlps(PhiLix)
                    PhiL_nphi = 2 * PhiLi + 1
                    if (PhiLi > 20) then
                        PhiL_nphi = 2 * nint(0.5d0 * PhiL_nphi/dPh) + 1
                    end if
                    PhiL_phi_dphi = 2.d0 * M_PI / PhiL_nphi
                    fac  = (PhiL_phi_dphi * PhiLi * dPh) * (dphi * L1 * dL) / ( (2. * M_PI) ** 4. ) * 0.25d0
                    if (phiIx /= 0) then
                        fac = fac * 2d0
                    end if
                    do PhiL_nphi_ix = -(PhiL_nphi-1)/2, (PhiL_nphi-1)/2
                        PhiL_phi = PhiL_phi_dphi * PhiL_nphi_ix
                        PhiLx = PhiLi * cos(PhiL_phi)
                        PhiLy = PhiLi * sin(PhiL_phi)
                        L3x = PhiLx - L1x
                        L3y = PhiLy - L1y
                        L3 = sqrt(L3x*L3x + L3y*L3y)
                        if ((L3 >= lminB) .AND. (L3 <= lmaxB)) then
                            L3i = nint(L3)
                            L4x = -Lx - L3x
                            L4y = -Ly - L3y
                            L4 = sqrt(L4x * L4x + L4y * L4y)
                             if ((L4 >= lminB) .AND. (L4 <= lmaxB)) then
                                 L4i = nint(L4)
                                 term1 = wf(kA, L1x, L2x, L1y, L2y, L1i, L2i, clttfid, cltefid, cleefid, &
                                         lmaxttfid, lmaxtefid, lmaxeefid )&
                                         * wf(kB, L3x, L4x, L3y, L4y, L3i, L4i, clttfid, cltefid, cleefid, &
                                                 lmaxttfid, lmaxtefid, lmaxeefid)&
                                         * wf(k13, L1x, L3x, L1y, L3y, L1i, L3i, cltt, clte, clee, &
                                                 lmaxtt, lmaxte, lmaxee)&
                                         * wf(k24, L2x, L4x, L2y, L4y, L2i, L4i, cltt, clte, clee, &
                                                 lmaxtt, lmaxte, lmaxee)&
                                         * fal1(L1i) * fal2(L2i) * fal3(L3i) * fal4(L4i)
                                 term2 = wf(kA, L1x, L2x, L1y, L2y, L1i, L2i, clttfid, cltefid, cleefid, &
                                           lmaxttfid, lmaxtefid, lmaxeefid)&
                                         * wf(kB, L4x, L3x, L4y, L3y, L4i, L3i, clttfid, cltefid, cleefid, &
                                           lmaxttfid, lmaxtefid, lmaxeefid)&
                                         * wf(k14, L1x, L3x, L1y, L3y, L1i, L3i, cltt, clte, clee, &
                                                 lmaxtt, lmaxte, lmaxee)&
                                         * wf(k23, L2x, L4x, L2y, L4y, L2i, L4i, cltt, clte, clee, &
                                                 lmaxtt, lmaxte, lmaxee)&
                                         * fal1(L1i) * fal2(L2i) * fal3(L4i) * fal4(L3i)
                                 n1L_jtp = n1L_jtp  + (term1 + term2) * fac * cl_kI(PhiLi)
                            end if
                        end if
                    end do
                end do
            end if
        end do
    end do
end