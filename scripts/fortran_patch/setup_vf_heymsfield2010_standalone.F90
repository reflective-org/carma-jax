!! Standalone Fortran driver for setupvf_heymsfield2010.
!! Kernel transcribed verbatim from setupvf_heymsfield2010.F90:44-86,
!! collapsed to a single column (NZ = scenario-provided) and single
!! group. Multi-altitude is supported so a realistic column can be
!! benched end-to-end against the JAX kernel.
!!
!! Input file (whitespace, free-form):
!!   NZ  NBIN
!!   t(1)        ... t(NZ)
!!   rhoa(1)     ... rhoa(NZ)
!!   zmet(1)     ... zmet(NZ)
!!   rmu(1)      ... rmu(NZ)
!!   r_wet(1,1)  ... r_wet(NBIN,1)    ... r_wet(1,NZ) ... r_wet(NBIN,NZ)
!!     (NZ * NBIN values, k outer / i inner, matches Fortran column-major
!!      view r_wet(i,k))
!!   rrat(1)     ... rrat(NBIN)
!!   rmass(1)    ... rmass(NBIN)
!!   arat(1)     ... arat(NBIN)
!! Output: real64[3 * NZ * NBIN]
!!   vf  : vf(i,k), i fastest, then k
!!   re  : re(i,k)
!!   bpm : bpm(i,k)

program setup_vf_heymsfield2010_standalone
  implicit none
  integer, parameter :: f = selected_real_kind(15, 307)

  character(len=512) :: in_path, out_path
  integer            :: unit_in, unit_out, ios, nz, nbin, i, k

  real(kind=f), allocatable :: t(:), rhoa(:), zmet(:), rmu(:)
  real(kind=f), allocatable :: r_wet(:, :), rrat(:), rmass(:), arat(:)
  real(kind=f), allocatable :: vf(:, :), re(:, :), bpm(:, :)
  real(kind=f)       :: rhoa_cgs, vg, rmfp, rkn, expon, x, dmax

  ! Constants — verbatim from carma_constants_mod.F90
  real(kind=f), parameter :: PI       = 3.14159265358979_f
  real(kind=f), parameter :: GRAV     = 980.6_f
  real(kind=f), parameter :: RGAS     = 8.31430e7_f
  real(kind=f), parameter :: WTMOL_A  = 28.966_f
  real(kind=f), parameter :: R_AIR    = RGAS / WTMOL_A
  real(kind=f), parameter :: POWMAX   = 706.0_f
  real(kind=f), parameter :: c0       = 0.35_f
  real(kind=f), parameter :: delta0   = 8.0_f

  if (command_argument_count() < 2) then
    write(0, '(A)') 'usage: setup_vf_heymsfield2010_standalone <in.txt> <out.bin>'
    call exit(1)
  end if
  call get_command_argument(1, in_path)
  call get_command_argument(2, out_path)

  open(newunit=unit_in, file=trim(in_path), action='read', status='old', iostat=ios)
  read(unit_in, *, iostat=ios) nz, nbin
  if (ios /= 0) then; write(0, '(A)') 'malformed header'; call exit(3); end if

  allocate(t(nz), rhoa(nz), zmet(nz), rmu(nz))
  allocate(r_wet(nbin, nz), rrat(nbin), rmass(nbin), arat(nbin))
  allocate(vf(nbin, nz), re(nbin, nz), bpm(nbin, nz))

  read(unit_in, *, iostat=ios) (t(k),    k = 1, nz)
  read(unit_in, *, iostat=ios) (rhoa(k), k = 1, nz)
  read(unit_in, *, iostat=ios) (zmet(k), k = 1, nz)
  read(unit_in, *, iostat=ios) (rmu(k),  k = 1, nz)
  read(unit_in, *, iostat=ios) ((r_wet(i, k), i = 1, nbin), k = 1, nz)
  read(unit_in, *, iostat=ios) (rrat(i),  i = 1, nbin)
  read(unit_in, *, iostat=ios) (rmass(i), i = 1, nbin)
  read(unit_in, *, iostat=ios) (arat(i),  i = 1, nbin)
  close(unit_in)

  do k = 1, nz
    rhoa_cgs = rhoa(k) / zmet(k)
    vg = sqrt(8._f / PI * R_AIR * t(k))
    rmfp = 2._f * rmu(k) / (rhoa_cgs * vg)

    do i = 1, nbin
      rkn = rmfp / (r_wet(i, k) * rrat(i))
      expon = -0.87_f / rkn
      expon = max(-POWMAX, expon)
      bpm(i, k) = 1._f + (1.246_f * rkn + 0.42_f * rkn * exp(expon))

      dmax = 2._f * r_wet(i, k) * rrat(i)

      x = (rhoa_cgs / (rmu(k)**2)) * &
           ((8._f * rmass(i) * GRAV) / (PI * (arat(i)**0.5_f)))
      x = x * bpm(i, k)

      re(i, k) = ((delta0**2) / 4._f) * &
                 (sqrt(1._f + (4._f * sqrt(x) / (delta0**2 * sqrt(c0)))) - 1._f)**2

      vf(i, k) = rmu(k) * re(i, k) / (rhoa_cgs * dmax)
    end do
  end do

  open(newunit=unit_out, file=trim(out_path), access='stream', form='unformatted', &
       action='write', status='replace', iostat=ios)
  write(unit_out) vf
  write(unit_out) re
  write(unit_out) bpm
  close(unit_out)

  deallocate(t, rhoa, zmet, rmu, r_wet, rrat, rmass, arat, vf, re, bpm)
end program setup_vf_heymsfield2010_standalone
