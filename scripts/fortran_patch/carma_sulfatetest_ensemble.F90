!! Ensemble-driven sulfate test — reads one scenario from argv[1],
!! writes one JSON output to argv[2]. Honors the contract in
!! carma-jax/docs/decisions/0025-fortran-orchestrator-design.md.
!!
!! Scenario line (whitespace-separated):
!!   T_K  p_hPa  rh_fraction  h2so4_pptv  aerosol_mu_nm  aerosol_sigma_g
!!
!! Output JSON keys:
!!   T_final, gc_h2so4_final, pc_final (NBIN-element array), nstep_ran, status
!!
!! Physics is identical to carma_sulfatetest.F90; only hard-coded
!! constants are replaced by the scenario values. Drop this file
!! into ../original-carma/CARMA/tests/ alongside carma_sulfatetest.F90
!! and add one line to tests/CMakeLists.txt:
!!
!!   create_standard_test(NAME sulfate_ensemble SOURCES atmosphere_mod.F90 \
!!                        carma_sulfatetest_ensemble.F90)
!!
!! Then rebuild — the binary appears at
!! ../original-carma/CARMA/build/carma/SULFATE_ENSEMBLE.exe.

program carma_sulfatetest_ensemble
  implicit none
  call test_sulfate_ensemble()
end program

subroutine test_sulfate_ensemble()
  use carma_precision_mod
  use carma_constants_mod
  use carma_enums_mod
  use carma_types_mod
  use carmaelement_mod
  use carmagroup_mod
  use carmagas_mod
  use carmastate_mod
  use carma_mod
  use atmosphere_mod

  implicit none

  integer, parameter        :: NZ           = 1
  integer, parameter        :: NZP1         = NZ+1
  integer, parameter        :: NELEM        = 1
  integer, parameter        :: NBIN         = 38
  integer, parameter        :: NGROUP       = 1
  integer, parameter        :: NSOLUTE      = 0
  integer, parameter        :: NGAS         = 2
  integer, parameter        :: NWAVE        = 0
  integer, parameter        :: LUNOPRT      = 6

  real(kind=f), parameter   :: dtime  = 1800._f
  real(kind=f), parameter   :: deltaz = 10000._f
  real(kind=f), parameter   :: zmin   = 17000._f
  integer, parameter        :: nstep  = 180000 / int(dtime)

  integer, parameter        :: I_H2SO4  = 1

  character(len=512)        :: scenario_path, output_path
  real(kind=f)              :: T_scen, p_scen_hPa, rh_scen
  real(kind=f)              :: h2so4_pptv, mu_nm, sigma_g
  integer                   :: unit_in, unit_out, ios

  type(carma_type), target            :: carma
  type(carma_type), pointer           :: carma_ptr
  type(carmastate_type)               :: cstate
  integer                             :: rc = 0

  real(kind=f), allocatable   :: zc(:), zl(:), p(:), pl(:), t(:), rho(:)
  real(kind=f), allocatable   :: mmr(:,:,:), mmr_gas(:,:), new_gas(:,:)
  real(kind=f), allocatable   :: satliq(:,:), satice(:,:)
  real(kind=f), allocatable   :: r(:), rmass(:)

  real(kind=f)          :: lat, lon
  integer               :: i, istep, igas, ielem, ibin
  integer               :: nsubsteps, lastsub
  real(kind=f)          :: nretries
  real(kind=f)          :: time
  real(kind=f)          :: rmin, rmrat, RHO_SULFATE
  real(kind=f)          :: log_r, log_mu_cm, log_sigma, norm, r_cm

  ! Phase 10.6: track substep schedule per outer step for prescribed-
  ! substep parity test. Each entry is the ntsubsteps Fortran's
  ! adaptive retry settled on for that outer step.
  integer(kind=8), allocatable :: nsubsteps_history(:)
  integer(kind=8), allocatable :: nretries_history(:)
  character(len=512)           :: schedule_path
  integer                      :: unit_sched

  if (command_argument_count() < 2) then
    write(0, '(A)') 'usage: <scenario_file> <output_file>'
    call exit(2)
  end if
  call get_command_argument(1, scenario_path)
  call get_command_argument(2, output_path)

  open(newunit=unit_in, file=trim(scenario_path), action='read', &
       status='old', iostat=ios)
  if (ios /= 0) then
    write(0, '(A, A)') 'cannot open scenario file: ', trim(scenario_path)
    call exit(3)
  end if
  read(unit_in, *, iostat=ios) T_scen, p_scen_hPa, rh_scen, &
                               h2so4_pptv, mu_nm, sigma_g
  close(unit_in)
  if (ios /= 0) then
    write(0, '(A)') 'malformed scenario file'
    call exit(4)
  end if

  allocate(zc(NZ), zl(NZP1), p(NZ), pl(NZP1), t(NZ), rho(NZ))
  allocate(mmr(NZ,NELEM,NBIN))
  allocate(mmr_gas(NZ,NGAS), new_gas(NZ,NGAS))
  allocate(satliq(NZ,NGAS), satice(NZ,NGAS))
  allocate(r(NBIN), rmass(NBIN))

  call CARMA_Create(carma, NBIN, NELEM, NGROUP, NSOLUTE, NGAS, NWAVE, rc, &
      LUNOPRT=LUNOPRT)
  if (rc /= 0) stop '*** CARMA_Create FAILED ***'
  carma_ptr => carma

  rmrat = 2._f
  rmin  = 2.e-8_f
  RHO_SULFATE = 1.923_f

  call CARMAGROUP_Create(carma, 1, 'sulfate', rmin, rmrat, I_SPHERE, 1._f, &
      .false., rc, irhswell=I_WTPCT_H2SO4, do_drydep=.true., &
      shortname='SULF', is_sulfate=.true.)
  if (rc /= 0) stop '*** CARMAGROUP_Create FAILED ***'

  call CARMAELEMENT_Create(carma, 1, 1, 'Sulfate', RHO_SULFATE, I_VOLATILE, &
      I_H2SO4, rc, shortname='SULF')
  if (rc /= 0) stop '*** CARMAELEMENT_Create FAILED ***'

  call CARMAGAS_Create(carma, 1, 'Water Vapor', WTMOL_H2O, &
      I_VAPRTN_H2O_MURPHY2005, I_GCOMP_H2O, rc, shortname='Q', &
      dgc_threshold=0.1_f, ds_threshold=0.1_f)
  if (rc /= 0) stop '*** CARMAGAS_Create FAILED ***'

  call CARMAGAS_Create(carma, 2, 'Sulpheric Acid', 98.078479_f, &
      I_VAPRTN_H2SO4_AYERS1980, I_GCOMP_H2SO4, rc, shortname='H2SO4', &
      dgc_threshold=0.1_f, ds_threshold=0.1_f)
  if (rc /= 0) stop '*** CARMAGAS_Create FAILED ***'

  call CARMA_AddGrowth(carma, 1, 2, rc)
  if (rc /= 0) stop '*** CARMA_AddGrowth FAILED ***'
  call CARMA_AddNucleation(carma, 1, 1, I_HOMNUC, 0._f, rc, igas=2)
  if (rc /= 0) stop '*** CARMA_AddNucleation FAILED ***'
  call CARMA_AddCoagulation(carma, 1, 1, 1, I_COLLEC_FUCHS, rc)
  if (rc /= 0) stop '*** CARMA_AddCoagulation FAILED ***'

  call CARMA_Initialize(carma, rc, do_grow=.true., do_coag=.true., &
      do_substep=.true., do_thermo=.true., maxretries=16, maxsubsteps=32, &
      dt_threshold=1._f, sulfnucl_method='ZhaoTurco')
  if (rc /= 0) stop '*** CARMA_Initialize FAILED ***'

  lat = -40.0_f
  lon = -105.0_f
  do i = 1, NZ
    zc(i) = zmin + (deltaz * (i - 0.5_f))
  end do
  call GetStandardAtmosphere(zc, p=p, t=t)
  do i = 1, NZP1
    zl(i) = zmin + ((i - 1) * deltaz)
  end do
  call GetStandardAtmosphere(zl, p=pl)

  ! --- Substitute scenario values ---
  t(1)  = T_scen
  p(1)  = p_scen_hPa * 100._f
  zl(1) = zc(1) - deltaz
  zl(2) = zc(1) + deltaz
  rho(1) = (p(1) * 10._f) / (R_AIR * t(1)) * (1e-3_f * 1e6_f)
  pl(1) = p(1) - (zl(1) - zc(1)) * rho(1) * (GRAV / 100._f)
  pl(2) = p(1) - (zl(2) - zc(1)) * rho(1) * (GRAV / 100._f)

  ! H2O from RH · p_vap(T) (Murphy-Koop-inspired analytic form)
  mmr_gas(:,1) = rh_scen * exp(54.842763_f - 6763.22_f/t(1) &
                               - 4.210_f*log(t(1)) + 0.000367_f*t(1)) &
                 * 18._f / (29._f * p(1))
  mmr_gas(:,2) = h2so4_pptv * 1.e-12_f * (98._f / 29._f)

  satliq(:,:)   = -1._f
  satice(:,:)   = -1._f
  mmr(:,:,:)    = 0._f

  ! Lognormal initial aerosol shape (mass-weighted), normalised so
  ! Σ_bin mmr(ibin) = 1e-18 g/g (total particle mass mixing ratio).
  ! Earlier version used `Σ mmr · rmass = 1e-18` which inflated the
  ! seed by factor ~Σ_bin(1/rmass) ≈ 1e22 — that was the source of
  ! the top-bin runaway. Fixed here.
  call CARMAGROUP_Get(carma, 1, rc, r=r, rmass=rmass)
  log_mu_cm = log(mu_nm * 1.e-7_f)
  log_sigma = log(sigma_g)
  norm = 0._f
  do ibin = 1, NBIN
    r_cm = r(ibin)
    log_r = log(r_cm)
    mmr(1,1,ibin) = exp(-0.5_f * ((log_r - log_mu_cm) / log_sigma)**2) &
                    / (r_cm * log_sigma * sqrt(2._f * PI)) &
                    * rmass(ibin)             ! mass-weighted shape
    norm = norm + mmr(1,1,ibin)
  end do
  if (norm > 0._f) then
    mmr(1,1,:) = mmr(1,1,:) * (1.e-18_f / norm)
  end if

  lastsub = 0
  ! Phase 10.6 schedule history: one entry per outer step.
  allocate(nsubsteps_history(nstep), nretries_history(nstep))
  nsubsteps_history(:) = 0
  nretries_history(:) = 0

  do istep = 1, nstep
    time = (istep - 1) * dtime
    call CARMASTATE_Create(cstate, carma_ptr, time, dtime, NZ, &
        I_CART, lat, lon, zc(:), zl(:), p(:), pl(:), t(:), rc, &
        told=t(:), qh2o=mmr_gas(1,:))
    if (rc /= 0) stop '*** CARMASTATE_Create FAILED ***'
    do ielem = 1, NELEM
      do ibin = 1, NBIN
        call CARMASTATE_SetBin(cstate, ielem, ibin, mmr(:,ielem,ibin), rc)
      end do
    end do
    new_gas = mmr_gas(:,:)
    do igas = 1, NGAS
      call CARMASTATE_SetGas(cstate, igas, new_gas(:,igas), rc, &
          mmr_old=mmr_gas(:,igas), &
          satice_old=satice(:,igas), satliq_old=satliq(:,igas))
    end do
    call CARMASTATE_Step(cstate, rc)
    if (rc /= 0) stop '*** CARMASTATE_Step FAILED ***'
    call CARMASTATE_Get(cstate, rc, nsubstep=nsubsteps, nretry=nretries)
    nsubsteps_history(istep) = int(nsubsteps, kind=8)
    nretries_history(istep)  = int(nretries, kind=8)
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
  end do

  ! Phase 10.6: write substep schedule to a sidecar binary file
  ! "<output_path>.schedule.bin" — 2 × nstep int64 (column-major):
  !   nsubsteps_history[1..nstep], nretries_history[1..nstep].
  schedule_path = trim(output_path) // '.schedule.bin'
  open(newunit=unit_sched, file=trim(schedule_path), &
       access='stream', form='unformatted', &
       action='write', status='replace', iostat=ios)
  if (ios /= 0) then
    write(0, '(A, A)') 'cannot write schedule: ', trim(schedule_path)
    call exit(6)
  end if
  write(unit_sched) nsubsteps_history
  write(unit_sched) nretries_history
  close(unit_sched)

  open(newunit=unit_out, file=trim(output_path), action='write', &
       status='replace', iostat=ios)
  if (ios /= 0) then
    write(0, '(A, A)') 'cannot write output: ', trim(output_path)
    call exit(5)
  end if
  write(unit_out, '(A)') '{'
  write(unit_out, '(A, ES16.8, A)') '  "T_final": ', t(1), ','
  write(unit_out, '(A, ES16.8, A)') '  "gc_h2so4_final": ', mmr_gas(1,2), ','
  write(unit_out, '(A)', advance='no') '  "pc_final": ['
  do ibin = 1, NBIN
    if (ibin > 1) write(unit_out, '(A)', advance='no') ', '
    write(unit_out, '(ES16.8)', advance='no') mmr(1,1,ibin)
  end do
  write(unit_out, '(A)') '],'
  write(unit_out, '(A, I0, A)') '  "nstep_ran": ', nstep, ','
  write(unit_out, '(A)') '  "status": "ok"'
  write(unit_out, '(A)') '}'
  close(unit_out)

  call CARMASTATE_Destroy(cstate, rc)
  call CARMA_Destroy(carma, rc)

  deallocate(zc, zl, p, pl, t, rho, mmr, mmr_gas, new_gas)
  deallocate(satliq, satice, r, rmass)
  deallocate(nsubsteps_history, nretries_history)

end subroutine test_sulfate_ensemble
