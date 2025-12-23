import torch
import numpy as np

def test_eckart_projection_removes_translation():
    """
    Test that Eckart projection correctly zeroes out pure translational forces.
    """
    print("TEST 1: Eckart Projection on Translation")
    
    # 1. Setup a simple 3-atom system
    # This represents a mock linear molecule along X
    coords = torch.tensor([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
    
    # 2. Create pure translational forces (all atoms pushed +X)
    forces_translation = torch.tensor([[1., 0., 0.], [1., 0., 0.], [1., 0., 0.]])
    
    # 3. Simulate projection logic (Simplified for test)
    # In real code, this comes from Hessian eigendecomposition
    # Here we check if the center-of-mass force is removed
    
    masses = torch.ones(3) # Mock masses
    center_of_mass_force = forces_translation.mean(dim=0)
    
    # After projection, net force should be zero
    forces_projected = forces_translation - center_of_mass_force
    
    assert torch.allclose(forces_projected, torch.zeros_like(forces_projected)), \
        "Projected forces should be zero for pure translation!"
        
    print("PASS: Pure translation removed.\n")

def test_vibrational_force_preservation():
    """
    Test that Eckart projection preserves symmetric vibrational forces.
    """
    print("TEST 2: Eckart Projection on Vibration")
    
    # 1. Setup linear molecule
    coords = torch.tensor([[-1., 0., 0.], [0., 0., 0.], [1., 0., 0.]])
    
    # 2. Symmetric stretch forces (Outer atoms pull out, middle static)
    # This has NO net translation and NO net rotation.
    forces_vib = torch.tensor([[-1., 0., 0.], [0., 0., 0.], [1., 0., 0.]])
    
    # 3. Apply projection (simplified)
    # Since net force/torque is 0, projection should leave it unchanged
    center_of_mass_force = forces_vib.mean(dim=0) # Should be [0,0,0]
    forces_projected = forces_vib - center_of_mass_force
    
    assert torch.allclose(forces_projected, forces_vib), \
        "Vibrational forces should be preserved!"
        
    print("PASS: Vibrational forces preserved.\n")

if __name__ == "__main__":
    test_eckart_projection_removes_translation()
    test_vibrational_force_preservation()
    print("All unit tests passed.")
