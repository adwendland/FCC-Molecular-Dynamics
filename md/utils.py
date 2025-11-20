def write_xyz(system, step, filename="traj.xyz"):
    """
    Write extended XYZ with cell information (OVITO compatible).
    """
    mode = "a" if step > 0 else "w"
    with open(filename, mode) as f:
        N = system.N
        Lx, Ly, Lz = system.box   # assuming box = [Lx, Ly, Lz]

        f.write(f"{N}\n")
        f.write(
            f"Step={step} "
            f'Lattice="{Lx} 0 0  0 {Ly} 0  0 0 {Lz}" '
            f'Properties=species:S:1:pos:R:3\n'
        )

        for i in range(N):
            x, y, z = system.pos[i]
            f.write(f"{system.symbol} {x:.8f} {y:.8f} {z:.8f}\n")
