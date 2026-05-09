!! Diagnostic-instrumented carma_nuc2test (Phase 11.4, "option B").
!!
!! Mirrors carma_sulfatetest_diagnostic.F90's pattern for the
!! `carma_nuc2test` config: 2 groups (Sulfate IN + Ice Crystal),
!! 3 elements (sulfate number, ice number, core mass), 1 gas
!! (water vapor), I_AERFREEZE + I_AF_MOHLER_2010 nucleation. The
!! diagnostic dumps per-step state + a probe that calls
!! `freezaerl_mohler2010` directly at end-of-step state and dumps
!! the resulting `cstate%f_rnuclg`. Lets us validate the JAX port in
!! its real upstream-subroutine dispatch context (not just the
!! standalone formula bench).
!!
!! Reads ONE scenario from argv[1], writes per-step dumps into argv[2].
!! Optional argv[3]: nstep_max (default 1000).
!! Optional argv[4]: kernel selector — "mohler" (default), "tabazadeh", or "koop".
!! Selects which freezaerl_* subroutine the AddNucleation flag wires
!! up and which the probe calls.
!!
!! Scenario line (whitespace-separated):
!!   T_K  p_hPa  rh_fraction  n_concentration_cm3  mu_radius_cm  sigma_g
!!   T_override  ssi_override  ssl_override  akelvin_over  akelvini_over  pconmax_over
!!
!! The first 6 fields drive the initial nuc2test column setup. The
!! second 6 (the *_override fields) are injected into cstate after
!! CARMASTATE_Step so the freezaerl_mohler probe runs with controlled
!! inputs instead of the post-step (nearly water-depleted) state.
!! This mirrors the standalone-Fortran bench's parameter coverage but
!! routes through the upstream subroutine's full dispatch wrapper.
!!
!! Output files (one set per outer step):
!!   substep_<NNNN>_pc.bin           shape (NBIN, NELEM)
!!   substep_<NNNN>_gc.bin           shape (NGAS,)
!!   substep_<NNNN>_t.bin            shape (NZ,)
!!   substep_<NNNN>_p.bin            shape (NZ,)
!!   substep_<NNNN>_rhoa.bin         shape (NZ,)
!!   substep_<NNNN>_zmet.bin         shape (NZ,)
!!   substep_<NNNN>_supsati.bin      shape (NZ, NGAS)
!!   substep_<NNNN>_supsatl.bin      shape (NZ, NGAS)
!!   substep_<NNNN>_akelvin.bin      shape (NZ, NGAS)
!!   substep_<NNNN>_akelvini.bin     shape (NZ, NGAS)
!!   substep_<NNNN>_r_wet.bin        shape (NZ, NBIN, NGROUP)
!!   substep_<NNNN>_pconmax.bin      shape (NZ, NGROUP)
!!   substep_<NNNN>_freezaerl_mohler_probe.bin
!!                                    shape (NBIN, NGROUP, NGROUP)
!!   substep_<NNNN>_r_bin.bin        shape (NBIN, NGROUP)  -- step 1 only (static)
!!   substep_<NNNN>_rmass_bin.bin    shape (NBIN, NGROUP)  -- step 1 only
!!   substep_<NNNN>_vol_bin.bin      shape (NBIN, NGROUP)  -- step 1 only
!!   substep_<NNNN>_rhosol.bin       shape (NSOLUTE,)      -- step 1 only
!!
!! Build: bash scripts/fortran_patch/apply_nuc2_diagnostic_patch.sh

program carma_nuc2test_diagnostic
  implicit none
  call test_nuc2_diagnostic()
end program

subroutine test_nuc2_diagnostic()
  use carma_precision_mod
  use carma_constants_mod
  use carma_enums_mod
  use carma_types_mod
  use carmaelement_mod
  use carmagroup_mod
  use carmagas_mod
  use carmasolute_mod
  use carmastate_mod
  use carma_mod
  use atmosphere_mod

  implicit none

  ! Mirror carma_nuc2test.F90 dimensions.
  integer, parameter :: NZ      = 1
  integer, parameter :: NZP1    = NZ + 1
  integer, parameter :: NELEM   = 3
  integer, parameter :: NBIN    = 16
  integer, parameter :: NGROUP  = 2
  integer, parameter :: NSOLUTE = 1
  integer, parameter :: NGAS    = 1
  integer, parameter :: NWAVE   = 0
  integer, parameter :: LUNOPRT = 6

  ! Element + composition labels.
  integer, parameter :: I_H2SO4 = 1
  integer, parameter :: I_ICE   = 2
  integer, parameter :: I_WATER = 3
  real(kind=f), parameter :: RHO_CN = 1.78_f

  type(carma_type), target  :: carma
  type(carma_type), pointer :: carma_ptr
  type(carmastate_type)     :: cstate
  integer                   :: rc = 0

  real(kind=f), allocatable :: zc(:), zl(:), p(:), pl(:), t(:), rho(:)
  real(kind=f), allocatable :: mmr(:,:,:), mmr_gas(:,:)
  real(kind=f), allocatable :: satliq(:,:), satice(:,:)
  real(kind=f), allocatable :: r(:), dr(:), rmass(:)

  real(kind=f) :: rmin, rmrat, deltaz, zmin
  real(kind=f) :: lat, lon, time, dtime
  real(kind=f) :: T_scen, p_scen_hPa, rh_scen
  real(kind=f) :: n_scen, mu_scen_cm, rsig_scen
  real(kind=f) :: r_cm, log_r, log_mu_cm, log_sigma, norm

  ! Scenario overrides for the Möhler probe (injected post-Step).
  real(kind=f) :: T_over, ssi_over, ssl_over, akelvin_over, akelvini_over, pconmax_over

  integer :: nstep, nstep_max
  integer :: istep, ielem, ibin, igas, igroup
  integer :: ios, n_dumped, nsubsteps
  integer :: kernel_flag    ! I_AF_MOHLER_2010 / I_AF_TABAZADEH_2000 / I_AF_KOOP_2000
  real(kind=f) :: nretries
  character(len=512) :: scen_path, out_dir, prefix, arg_buf
  character(len=32)  :: kernel_name

  if (command_argument_count() < 2) then
    write(0, '(A)') 'usage: <scenario_file> <output_dir> [nstep_max] [kernel]'
    write(0, '(A)') '  kernel: "mohler" (default), "tabazadeh", or "koop"'
    call exit(1)
  end if
  call get_command_argument(1, scen_path)
  call get_command_argument(2, out_dir)
  nstep_max = 0
  if (command_argument_count() >= 3) then
    call get_command_argument(3, arg_buf)
    read(arg_buf, *, iostat=ios) nstep_max
    if (ios /= 0) nstep_max = 0
  end if
  kernel_name = "mohler"
  if (command_argument_count() >= 4) then
    call get_command_argument(4, kernel_name)
  end if
  select case (trim(kernel_name))
    case ("mohler", "MOHLER")
      kernel_flag = I_AF_MOHLER_2010
    case ("tabazadeh", "TABAZADEH")
      kernel_flag = I_AF_TABAZADEH_2000
    case ("koop", "KOOP")
      kernel_flag = I_AF_KOOP_2000
    case default
      write(0, '(A, A)') 'unknown kernel: ', trim(kernel_name)
      call exit(1)
  end select

  ! Read scenario (whitespace, free-form): T p rh n mu_cm sigma_g
  ! followed by the 6 override fields for the Möhler probe.
  open(newunit=ios, file=trim(scen_path), action='read', status='old')
  read(ios, *) T_scen, p_scen_hPa, rh_scen, n_scen, mu_scen_cm, rsig_scen, &
               T_over, ssi_over, ssl_over, akelvin_over, akelvini_over, pconmax_over
  close(ios)

  ! Atmosphere setup mirrors carma_nuc2test.F90.
  dtime  = 1.0_f
  deltaz = 100.0_f
  zmin   = 3000.0_f

  allocate(zc(NZ), zl(NZP1), p(NZ), pl(NZP1), t(NZ), rho(NZ))
  allocate(mmr(NZ, NELEM, NBIN), mmr_gas(NZ, NGAS))
  allocate(satliq(NZ, NGAS), satice(NZ, NGAS))
  allocate(r(NBIN), dr(NBIN), rmass(NBIN))

  ! CARMA model.
  call CARMA_Create(carma, NBIN, NELEM, NGROUP, NSOLUTE, NGAS, NWAVE, rc, &
                    LUNOPRT=LUNOPRT)
  if (rc /= 0) call exit(2)
  carma_ptr => carma

  ! Sulfate IN group: 16 bins from r=1e-7 cm = 1 nm with rmrat=4.
  call CARMAGROUP_Create(carma, 1, "Sulfate IN", 1.e-7_f, 4._f, I_SPHERE, 1._f, &
                         .false., rc, do_wetdep=.true., do_drydep=.false., &
                         solfac=0.3_f, scavcoef=0.1_f, shortname="CRIN", do_mie=.false.)
  if (rc /= 0) call exit(3)

  ! Ice Crystal group: rmin=5e-5 cm = 500 nm, rmrat=4, ice rho.
  call CARMAGROUP_Create(carma, 2, "Ice Crystal", 5.e-5_f, 4.0_f, I_SPHERE, 3._f, &
                         .true., rc, do_wetdep=.true., do_drydep=.false., &
                         solfac=0.3_f, scavcoef=0.1_f, shortname="CRICE", do_mie=.false.)
  if (rc /= 0) call exit(4)

  ! Elements: 1 = sulfate number (involatile, H2SO4 composition),
  !           2 = ice number (volatile, ice composition),
  !           3 = sulfate core mass on ice group.
  call CARMAELEMENT_Create(carma, 1, 1, "Sulfate IN", RHO_CN, I_INVOLATILE, &
                           I_H2SO4, rc, shortname="CRIN", isolute=1)
  if (rc /= 0) call exit(5)
  call CARMAELEMENT_Create(carma, 2, 2, "Ice Crystal", RHO_I, I_VOLATILE, &
                           I_ICE, rc, shortname="CRICE")
  if (rc /= 0) call exit(6)
  call CARMAELEMENT_Create(carma, 3, 2, "Core Mass", RHO_CN, I_COREMASS, &
                           I_H2SO4, rc, shortname="CRCORE", isolute=1)
  if (rc /= 0) call exit(7)

  ! Solute and gas.
  call CARMASOLUTE_Create(carma, 1, "Sulfuric Acid", 2, 98._f, 1.38_f, rc)
  if (rc /= 0) call exit(8)
  call CARMAGAS_Create(carma, 1, "Water Vapor", WTMOL_H2O, &
                       I_VAPRTN_H2O_MURPHY2005, I_GCOMP_H2O, rc, shortname='Q')
  if (rc /= 0) call exit(9)

  ! Growth + nucleation: Möhler 2010 only.
  call CARMA_AddGrowth(carma, 2, 1, rc)
  if (rc /= 0) call exit(10)
  call CARMA_AddNucleation(carma, 1, 3, I_AERFREEZE + kernel_flag, &
                           0._f, rc, igas=1, ievp2elem=1)
  if (rc /= 0) call exit(11)
  call CARMA_Initialize(carma, rc, do_grow=.true.)
  if (rc /= 0) call exit(12)

  ! Atmosphere column.
  lat = -40.0_f; lon = -105.0_f
  do ibin = 1, NZ
    zc(ibin) = zmin + (deltaz * (ibin - 0.5_f))
  end do
  call GetStandardAtmosphere(zc, p=p, t=t)
  do ibin = 1, NZP1
    zl(ibin) = zmin + ((ibin - 1) * deltaz)
  end do
  call GetStandardAtmosphere(zl, p=pl)

  ! Override scenario T, p (hPa → CGS dyne/cm² done by atmosphere helpers).
  t(1) = T_scen
  p(1) = p_scen_hPa * 1.e3_f      ! hPa → dyne/cm² (×100 Pa/hPa × 10 dyne/cm²/Pa)
  rho(1) = p(1) / (RGAS * t(1) / WTMOL_AIR)
  pl(1) = p(1) + 0.5_f * (zl(2) - zc(1)) * rho(1) * (GRAV / 100._f)
  pl(2) = p(1) - 0.5_f * (zc(1) - zl(1)) * rho(1) * (GRAV / 100._f)

  ! Initial gas: H2O at rh × Murphy-Koop saturation.
  mmr_gas(:,1) = rh_scen * exp(54.842763_f - 6763.22_f / t(1) &
                                - 4.210_f * log(t(1)) + 0.000367_f * t(1)) &
                 * 18._f / (29._f * p(1))

  satliq(:,:) = -1._f
  satice(:,:) = -1._f
  mmr(:,:,:) = 0._f

  ! Initial Sulfate IN aerosol: lognormal in ELEMENT 1 (number).
  call CARMAGROUP_Get(carma, 1, rc, r=r, rmass=rmass)
  log_mu_cm = log(mu_scen_cm)
  log_sigma = log(rsig_scen)
  norm = 0._f
  do ibin = 1, NBIN
    r_cm = r(ibin)
    log_r = log(r_cm)
    mmr(1, 1, ibin) = exp(-0.5_f * ((log_r - log_mu_cm) / log_sigma)**2) &
                      / (r_cm * log_sigma * sqrt(2._f * PI)) &
                      * rmass(ibin)
    norm = norm + mmr(1, 1, ibin)
  end do
  if (norm > 0._f) mmr(1, 1, :) = mmr(1, 1, :) * (n_scen * rmass(1) / norm)

  if (nstep_max > 0) then
    nstep = nstep_max
  else
    nstep = 1000
  end if
  n_dumped = 0

  do istep = 1, nstep
    time = (istep - 1) * dtime
    call CARMASTATE_Create(cstate, carma_ptr, time, dtime, NZ, &
        I_CART, lat, lon, zc(:), zl(:), p(:), pl(:), t(:), rc, &
        told=t(:), qh2o=mmr_gas(1,:))
    if (rc /= 0) call exit(20)
    do ielem = 1, NELEM
      do ibin = 1, NBIN
        call CARMASTATE_SetBin(cstate, ielem, ibin, mmr(:,ielem,ibin), rc)
      end do
    end do
    do igas = 1, NGAS
      call CARMASTATE_SetGas(cstate, igas, mmr_gas(:,igas), rc, &
          mmr_old=mmr_gas(:,igas), &
          satice_old=satice(:,igas), satliq_old=satliq(:,igas))
    end do
    call CARMASTATE_Step(cstate, rc)
    if (rc /= 0) call exit(21)
    call CARMASTATE_Get(cstate, rc, nsubstep=nsubsteps, nretry=nretries)
    call CARMASTATE_GetState(cstate, rc, t=t(:))
    do ielem = 1, NELEM
      do ibin = 1, NBIN
        call CARMASTATE_GetBin(cstate, ielem, ibin, mmr(:,ielem,ibin), rc)
      end do
    end do
    do igas = 1, NGAS
      call CARMASTATE_GetGas(cstate, igas, mmr_gas(:,igas), rc, &
          satliq=satliq(:,igas), satice=satice(:,igas))
    end do

    ! Build prefix: <out_dir>/substep_<NNNN>_
    call build_prefix(out_dir, istep, prefix)
    call dump_step(prefix, cstate, istep)
    n_dumped = n_dumped + 1
  end do

  call CARMASTATE_Destroy(cstate, rc)
  call CARMA_Destroy(carma, rc)

  deallocate(zc, zl, p, pl, t, rho, mmr, mmr_gas, satliq, satice, r, dr, rmass)

contains

  subroutine build_prefix(dir, n, out)
    character(len=*), intent(in)  :: dir
    integer,           intent(in) :: n
    character(len=*), intent(out) :: out
    character(len=8)              :: numbuf
    write(numbuf, '(I4.4)') n
    out = trim(dir) // '/substep_' // trim(numbuf) // '_'
  end subroutine build_prefix


  subroutine dump_step(prefix, cs, istep_loc)
    character(len=*), intent(in)         :: prefix
    type(carmastate_type), intent(inout) :: cs
    integer, intent(in)                  :: istep_loc

    real(kind=f), allocatable :: pc_2d(:,:)
    real(kind=f), allocatable :: r_bin_2d(:,:), rmass_bin_2d(:,:)
    real(kind=f), allocatable :: vol_bin_2d(:,:)
    real(kind=f), allocatable :: rhosol_arr(:)
    integer :: rc_loc, ielem_l, ibin_l, ig_l, isol_l, kk
    integer :: nbin_l, ngroup_l

    if (.not. allocated(cs%f_pc) .or. .not. allocated(cs%f_gc)) return

    ! State arrays.
    nbin_l   = NBIN
    ngroup_l = NGROUP
    allocate(pc_2d(nbin_l, NELEM))
    do ielem_l = 1, NELEM
      do ibin_l = 1, nbin_l
        pc_2d(ibin_l, ielem_l) = cs%f_pc(1, ibin_l, ielem_l)
      end do
    end do
    call dump_alloc_2d(prefix, 'pc',   pc_2d)
    deallocate(pc_2d)
    call dump_alloc_1d(prefix, 'gc',   cs%f_gc(1, :))
    call dump_alloc_1d(prefix, 't',    cs%f_t)
    call dump_alloc_1d(prefix, 'p',    cs%f_p)
    call dump_alloc_1d(prefix, 'rhoa', cs%f_rhoa)
    call dump_alloc_1d(prefix, 'zmet', cs%f_zmet)

    if (allocated(cs%f_supsati))   call dump_alloc_2d(prefix, 'supsati',  cs%f_supsati)
    if (allocated(cs%f_supsatl))   call dump_alloc_2d(prefix, 'supsatl',  cs%f_supsatl)
    if (allocated(cs%f_akelvin))   call dump_alloc_2d(prefix, 'akelvin',  cs%f_akelvin)
    if (allocated(cs%f_akelvini))  call dump_alloc_2d(prefix, 'akelvini', cs%f_akelvini)
    if (allocated(cs%f_r_wet))     call dump_alloc_3d(prefix, 'r_wet',    cs%f_r_wet)
    if (allocated(cs%f_pconmax))   call dump_alloc_2d(prefix, 'pconmax',  cs%f_pconmax)

    ! Probe: inject scenario overrides into cstate (so the kernel runs
    ! with controlled inputs through the full dispatch wrapper, rather
    ! than whatever post-Step state remains), zero rnuclg, call the
    ! selected freezaerl_*, dump.
    if (allocated(cs%f_rnuclg) .and. allocated(cs%f_supsati) .and. &
        allocated(cs%f_supsatl) .and. allocated(cs%f_akelvin) .and. &
        allocated(cs%f_akelvini) .and. allocated(cs%f_pconmax) .and. &
        allocated(cs%f_t)) then
      cs%f_t(1)             = T_over
      cs%f_supsati(1, 1)    = ssi_over
      cs%f_supsatl(1, 1)    = ssl_over
      cs%f_akelvin(1, 1)    = akelvin_over
      cs%f_akelvini(1, 1)   = akelvini_over
      cs%f_pconmax(1, 1)    = pconmax_over
      cs%f_rnuclg(:,:,:) = 0._f
      select case (trim(kernel_name))
        case ("mohler", "MOHLER")
          call freezaerl_mohler2010(carma, cs, 1, rc_loc)
          call dump_alloc_3d(prefix, 'freezaerl_mohler_probe', cs%f_rnuclg)
        case ("tabazadeh", "TABAZADEH")
          call freezaerl_tabazadeh2000(carma, cs, 1, rc_loc)
          call dump_alloc_3d(prefix, 'freezaerl_tabazadeh_probe', cs%f_rnuclg)
        case ("koop", "KOOP")
          call freezaerl_koop2000(carma, cs, 1, rc_loc)
          call dump_alloc_3d(prefix, 'freezaerl_koop_probe', cs%f_rnuclg)
      end select
    end if

    ! Step-1 statics: bin radii / masses / volumes / solute densities.
    if (istep_loc == 1) then
      allocate(r_bin_2d(nbin_l, ngroup_l), rmass_bin_2d(nbin_l, ngroup_l), &
               vol_bin_2d(nbin_l, ngroup_l))
      do ig_l = 1, ngroup_l
        do ibin_l = 1, nbin_l
          r_bin_2d(ibin_l, ig_l)     = carma%f_group(ig_l)%f_r(ibin_l)
          rmass_bin_2d(ibin_l, ig_l) = carma%f_group(ig_l)%f_rmass(ibin_l)
          vol_bin_2d(ibin_l, ig_l)   = carma%f_group(ig_l)%f_vol(ibin_l)
        end do
      end do
      call dump_alloc_2d(prefix, 'r_bin',     r_bin_2d)
      call dump_alloc_2d(prefix, 'rmass_bin', rmass_bin_2d)
      call dump_alloc_2d(prefix, 'vol_bin',   vol_bin_2d)
      deallocate(r_bin_2d, rmass_bin_2d, vol_bin_2d)
      allocate(rhosol_arr(NSOLUTE))
      do isol_l = 1, NSOLUTE
        rhosol_arr(isol_l) = carma%f_solute(isol_l)%f_rho
      end do
      call dump_alloc_1d(prefix, 'rhosol', rhosol_arr)
      deallocate(rhosol_arr)
    end if
  end subroutine dump_step


  subroutine dump_alloc_1d(prefix, name, arr)
    character(len=*), intent(in)         :: prefix, name
    real(kind=f), intent(in)             :: arr(:)
    character(len=512)                   :: path
    integer                              :: u, ios_l
    path = trim(prefix) // trim(name) // '.bin'
    open(newunit=u, file=trim(path), access='stream', &
         status='replace', iostat=ios_l)
    if (ios_l /= 0) return
    write(u) arr
    close(u)
  end subroutine dump_alloc_1d


  subroutine dump_alloc_2d(prefix, name, arr)
    character(len=*), intent(in)         :: prefix, name
    real(kind=f), intent(in)             :: arr(:,:)
    character(len=512)                   :: path
    integer                              :: u, ios_l
    path = trim(prefix) // trim(name) // '.bin'
    open(newunit=u, file=trim(path), access='stream', &
         status='replace', iostat=ios_l)
    if (ios_l /= 0) return
    write(u) arr
    close(u)
  end subroutine dump_alloc_2d


  subroutine dump_alloc_3d(prefix, name, arr)
    character(len=*), intent(in)         :: prefix, name
    real(kind=f), intent(in)             :: arr(:,:,:)
    character(len=512)                   :: path
    integer                              :: u, ios_l
    path = trim(prefix) // trim(name) // '.bin'
    open(newunit=u, file=trim(path), access='stream', &
         status='replace', iostat=ios_l)
    if (ios_l /= 0) return
    write(u) arr
    close(u)
  end subroutine dump_alloc_3d
end subroutine test_nuc2_diagnostic
