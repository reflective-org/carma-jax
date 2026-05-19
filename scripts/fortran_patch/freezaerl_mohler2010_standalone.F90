!! Standalone Fortran driver for freezaerl_mohler2010.
!!
!! Reads scenario inputs from a text file given as argv[1], computes
!! the per-bin nucleation loss rate via the Mohler 2010 formula
!! transcribed verbatim from
!! `original-carma/CARMA/source/base/freezaerl_mohler2010.F90:79-180`,
!! and writes the rnuclg array as a Fortran unformatted-stream binary
!! file given as argv[2].
!!
!! This is built without any CARMA dependency — pure gfortran numerical
!! kernel — so we can bench JAX against actual compiled-Fortran output
!! at the formula level. The carma-state dispatch around the kernel
!! (group / element / nucleation-process gating) is the caller's job
!! and not part of this comparison.
!!
!! Input file format (whitespace-separated, free-form):
!!   T  ssi  ssl  akelvin  akelvini  rhosol  pconmax  NBIN
!!   r(1) r(2) ... r(NBIN)
!!   vol(1) vol(2) ... vol(NBIN)
!!
!! Output file format (Fortran unformatted stream):
!!   real64[NBIN]   rnuclg
!!
!! All values in CGS to match CARMA convention.

program freezaerl_mohler2010_standalone
  implicit none
  integer, parameter   :: f = selected_real_kind(15, 307)

  character(len=512)   :: in_path, out_path
  integer              :: unit_in, unit_out, ios, nbin, ibin

  real(kind=f)         :: t, ssi_in, ssl_in, akelvin, akelvini
  real(kind=f)         :: rhosol, pconmax
  real(kind=f), allocatable :: r(:), vol(:), rnuclg(:)

  ! Constants matching original-carma/CARMA/source/base/carma_constants_mod.F90
  real(kind=f), parameter :: RHO_W   = 1.0_f
  real(kind=f), parameter :: SMALL_PC = 1.0e-50_f
  real(kind=f), parameter :: FEW_PC   = SMALL_PC * 1.0e6_f
  real(kind=f), parameter :: SIFREEZE = 0.3_f

  ! Scratch
  real(kind=f) :: ssi, ssl, fkelvi, fkelv, rlogj, rjj, aw
  real(kind=f) :: contl, conth, h2so4m, wt, volrat

  if (command_argument_count() < 2) then
    write(0, '(A)') 'usage: freezaerl_mohler2010_standalone <in.txt> <out.bin>'
    call exit(1)
  end if
  call get_command_argument(1, in_path)
  call get_command_argument(2, out_path)

  open(newunit=unit_in, file=trim(in_path), action='read', status='old', iostat=ios)
  if (ios /= 0) then
    write(0, '(A, A)') 'cannot open input: ', trim(in_path)
    call exit(2)
  end if
  read(unit_in, *, iostat=ios) t, ssi_in, ssl_in, akelvin, akelvini, rhosol, pconmax, nbin
  if (ios /= 0) then
    write(0, '(A)') 'malformed scalar header'
    call exit(3)
  end if
  allocate(r(nbin), vol(nbin), rnuclg(nbin))
  read(unit_in, *, iostat=ios) (r(ibin),   ibin = 1, nbin)
  read(unit_in, *, iostat=ios) (vol(ibin), ibin = 1, nbin)
  close(unit_in)
  if (ios /= 0) then
    write(0, '(A)') 'malformed bin arrays'
    call exit(4)
  end if

  rnuclg(:) = 0.0_f

  ! T <= 240 gate (freezaerl_mohler2010.F90:82)
  if (t <= 240.0_f .and. pconmax > FEW_PC) then
    do ibin = 1, nbin
      ssi = ssi_in
      ssl = ssl_in
      fkelvi = exp(akelvini / r(ibin))
      ssi = ssi / fkelvi
      if (ssi > SIFREEZE) then
        rlogj = 97.973292_f - 154.67476_f * (ssi + 1._f) &
              - 0.84952712_f * t + 1.0049467_f * (ssi + 1._f) * t
        rjj = 10._f ** rlogj
        ssl = max(-1.0_f, min(0._f, ssl))
        aw  = 1._f + ssl
        fkelv = exp(akelvin / r(ibin))
        aw = aw / fkelv

        if (aw < 0.05_f) then
          contl =  12.37208932_f  * (aw**(-0.16125516114_f)) - 30.490657554_f * aw - 2.1133114241_f
          conth =  13.455394705_f * (aw**(-0.1921312255_f))  - 34.285174604_f * aw - 1.7620073078_f
        else if (aw <= 0.85_f) then
          contl =  11.820654354_f * (aw**(-0.20786404244_f)) - 4.807306373_f  * aw - 5.1727540348_f
          conth =  12.891938068_f * (aw**(-0.23233847708_f)) - 6.4261237757_f * aw - 4.9005471319_f
        else
          contl = -180.06541028_f * (aw**(-0.38601102592_f)) - 93.317846778_f * aw + 273.88132245_f
          conth = -176.95814097_f * (aw**(-0.36257048154_f)) - 90.469744201_f * aw + 267.45509988_f
        end if

        h2so4m = contl + ((conth - contl) * (t - 190._f) / 70._f)
        wt = (98.0_f * h2so4m) / (1000._f + 98._f * h2so4m)
        wt = max(0._f, min(1._f, wt))
        wt = 100._f * wt

        if (wt <= 0._f) then
          volrat = 1.e10_f
        else
          volrat = rhosol / RHO_W * ((100._f - wt) / wt) + 1._f
        end if

        rnuclg(ibin) = min(1.e20_f, rjj * volrat * vol(ibin))
      end if
    end do
  end if

  open(newunit=unit_out, file=trim(out_path), access='stream', form='unformatted', &
       action='write', status='replace', iostat=ios)
  if (ios /= 0) then
    write(0, '(A, A)') 'cannot open output: ', trim(out_path)
    call exit(5)
  end if
  write(unit_out) rnuclg
  close(unit_out)

  deallocate(r, vol, rnuclg)
end program freezaerl_mohler2010_standalone
