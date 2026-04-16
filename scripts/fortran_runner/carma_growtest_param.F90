!! Parameterized CARMA growth test (24 bins).
!! Reads T, p, gas_mmr, N0, dtime, nstep from namelist.
!! Writes output to carma_growtest_param.txt.

program carma_growtest_param
  implicit none
  write(*,*) "Parameterized Growth Test"
  call test_growth()
  write(*,*) "Done"
end program


subroutine test_growth()
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

  integer, parameter    :: NX           = 1
  integer, parameter    :: NY           = 1
  integer, parameter    :: NZ           = 1
  integer, parameter    :: NZP1         = NZ+1
  integer, parameter    :: NELEM        = 1
  integer, parameter    :: NBIN         = 24
  integer, parameter    :: NGROUP       = 1
  integer, parameter    :: NSOLUTE      = 0
  integer, parameter    :: NGAS         = 1
  integer, parameter    :: NWAVE        = 0

  integer, parameter        :: I_H2O        = 1

  ! Parameters from namelist
  real(kind=f) :: param_T, param_p, param_gas_mmr, param_N0, param_dtime
  integer      :: param_nstep
  namelist /grow_params/ param_T, param_p, param_gas_mmr, param_N0, param_dtime, param_nstep

  real(kind=f), parameter   :: deltaz = 100._f

  type(carma_type), target  :: carma
  type(carma_type), pointer :: carma_ptr
  type(carmastate_type)     :: cstate
  integer                   :: rc = 0

  real(kind=f)  :: zc(NZ), zl(NZP1), p(NZ), pl(NZP1), t(NZ)
  real(kind=f)  :: mmr(NZ,NELEM,NBIN), mmr_gas(NZ,NGAS)
  real(kind=f)  :: lat, lon, rho_air, time, t_orig, rlheat(NZ)
  real(kind=f)  :: satliq(NZ,NGAS), satice(NZ,NGAS)

  integer       :: i, istep, ielem, ibin, igas
  integer, parameter :: lun = 42

  real(kind=f)  :: rmin, rmrat
  real(kind=f)  :: r(NBIN), dr(NBIN), rmass(NBIN)

  ! Defaults (TTL conditions)
  param_T = 190._f
  param_p = 9000._f
  param_gas_mmr = 3.5e-6_f
  param_N0 = 0.1_f
  param_dtime = 100._f
  param_nstep = 50

  ! Read namelist
  open(unit=10, file="grow_params.nml", status="old", iostat=rc)
  if (rc == 0) then
    read(10, nml=grow_params, iostat=rc)
    close(10)
  end if
  rc = 0

  open(unit=lun, file="carma_growtest_param.txt", status="unknown")

  call CARMA_Create(carma, NBIN, NELEM, NGROUP, NSOLUTE, NGAS, NWAVE, rc, LUNOPRT=6)
  carma_ptr => carma

  rmrat = 2._f
  rmin = 1e-4_f

  call CARMAGROUP_Create(carma, 1, "Ice Crystal", rmin, rmrat, I_SPHERE, 1._f, .TRUE., rc)
  call CARMAELEMENT_Create(carma, 1, 1, "Ice Crystal", RHO_I, I_VOLATILE, I_H2O, rc)
  call CARMAGAS_Create(carma, 1, "Water Vapor", WTMOL_H2O, I_VAPRTN_H2O_MURPHY2005, I_GCOMP_H2O, rc)
  call CARMA_AddGrowth(carma, 1, 1, rc)
  call CARMA_Initialize(carma, rc, do_grow=.true., do_thermo=.true.)

  lat = -40.0_f
  lon = -105.0_f

  ! Setup atmosphere from parameters
  t(1) = param_T
  p(1) = param_p
  zc(1) = 17000._f
  zl(1) = zc(1) - deltaz
  zl(2) = zc(1) + deltaz
  rho_air = (p(1) * 10._f) / (R_AIR * t(1)) * (1e-3_f * 1e6_f)
  pl(1) = p(1) - (zl(1) - zc(1)) * rho_air * (GRAV / 100._f)
  pl(2) = p(1) - (zl(2) - zc(1)) * rho_air * (GRAV / 100._f)

  ! Initial conditions
  call CARMAGROUP_Get(carma, 1, rc, rmass=rmass, r=r, dr=dr)

  mmr(:,:,:) = 0._f
  mmr(1,1,1) = (param_N0 * rmass(1) * (1e-3_f * 1e6_f)) / rho_air
  mmr_gas(:,:) = param_gas_mmr

  t_orig = t(1)

  ! Write header: NBIN, nstep, T, p
  write(lun,'(i4,1x,i6,1x,e15.8,1x,e15.8)') NBIN, param_nstep, t(1), p(1)

  ! Write bin structure
  do i = 1, NBIN
    write(lun,'(i3,1x,e15.8,1x,e15.8)') i, r(i), rmass(i)
  end do

  ! Write initial state
  write(lun,'(a)') "0"
  write(lun,'(e15.8,1x,e15.8)') 0._f, 0._f
  do i = 1, NBIN
    write(lun,'(i3,1x,e15.8)') i, mmr(1,1,i)
  end do
  write(lun,'(e15.8,1x,e15.8,1x,e15.8)') mmr_gas(1,1), 0._f, 0._f

  ! Time integration
  do istep = 1, param_nstep
    time = (istep - 1) * param_dtime

    call CARMASTATE_Create(cstate, carma_ptr, time, param_dtime, NZ, &
                           I_CART, lat, lon, zc, zl, p, pl, t, rc)

    do ielem = 1, NELEM
      do ibin = 1, NBIN
        call CARMASTATE_SetBin(cstate, ielem, ibin, mmr(:,ielem,ibin), rc)
      end do
    end do

    do igas = 1, NGAS
      call CARMASTATE_SetGas(cstate, igas, mmr_gas(:,igas), rc)
    end do

    call CARMASTATE_Step(cstate, rc)

    do ielem = 1, NELEM
      do ibin = 1, NBIN
        call CARMASTATE_GetBin(cstate, ielem, ibin, mmr(:,ielem,ibin), rc)
      end do
    end do

    do igas = 1, NGAS
      call CARMASTATE_GetGas(cstate, igas, mmr_gas(:,igas), rc, &
                             satliq=satliq(:,igas), satice=satice(:,igas))
    end do

    call CARMASTATE_GetState(cstate, rc, t=t, rlheat=rlheat)

    ! Write timestep
    write(lun,'(f12.1)') istep*param_dtime
    write(lun,'(e15.8,1x,e15.8)') t(1) - t_orig, rlheat(1)
    do i = 1, NBIN
      write(lun,'(i3,1x,e15.8)') i, mmr(1,1,i)
    end do
    write(lun,'(e15.8,1x,e15.8,1x,e15.8)') mmr_gas(1,1), satliq(1,1), satice(1,1)
  end do

  call CARMASTATE_Destroy(cstate, rc)
  close(unit=lun)
  call CARMA_Destroy(carma, rc)

end subroutine
