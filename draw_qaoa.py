import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set publication LaTeX math typography
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['mathtext.fontset'] = 'cm'

def generate_publication_qaoa_circuit(n_qubits, xy_mixer=False, save_path=None):
    """
    Generates an IEEE/APS style publication-ready quantum circuit diagram
    for Warm-Start QAOA across variable qubit counts N.
    """
    fig_height = max(3.5, n_qubits * 0.85 + 1.2)
    wire_y = [n_qubits - 1 - i for i in range(n_qubits)]
    
    # Palette
    c_wire = '#1F2937'        # Dark charcoal
    c_prep = '#1E3A8A'        # Deep royal blue
    c_cost_single = '#0284C7' # Azure blue
    c_cost_pair = '#6D28D9'   # Rich violet
    c_mixer = '#B91C1C' if xy_mixer else '#047857' # Red-orange (XY) / Emerald (WS-X)
    c_meas = '#374151'        # Slate gray
    c_barrier = '#9CA3AF'     # Neutral gray
    
    x_wire_start = 0.5
    
    # --- Dynamic Layout Geometry Calculation ---
    x_prep = 1.3
    x_barrier1 = x_prep + 1.1
    
    x_cost_single = x_barrier1 + 1.1
    rzz_pairs = [(i, i + 1) for i in range(n_qubits - 1)]
    if n_qubits > 2:
        rzz_pairs.append((0, n_qubits - 1))
    
    x_rzz_start = x_cost_single + 1.1
    x_barrier2 = x_rzz_start + len(rzz_pairs) * 1.25 + 0.1
    
    x_mixer_start = x_barrier2 + 1.1
    if not xy_mixer:
        x_barrier3 = x_mixer_start + 3.1
        mixer_title = "3. WS-X Mixer Phase $U(B_{WS}, \\beta)$"
    else:
        x_mixer_end = x_mixer_start + (n_qubits - 1) * 1.1 * 2 + 0.4
        x_barrier3 = x_mixer_end
        mixer_title = "3. XY Mixer Phase $U(B_{XY}, \\beta)$"
        
    # Set readout position and terminate wire dynamic distance after measurement
    x_meas = x_barrier3 + 1.0
    x_wire_end = x_meas + 0.8
    
    fig_width = max(10.0, x_wire_end * 0.95)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)
    ax.set_aspect('equal')
    ax.axis('off')

    def draw_box_gate(x, y, text, color, width=0.85, height=0.55, text_color='white', fontsize=8.5):
        rect = patches.FancyBboxPatch(
            (x - width/2, y - height/2), width, height,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            facecolor=color, edgecolor='#111827', linewidth=1.1, zorder=3
        )
        ax.add_patch(rect)
        ax.text(x, y, text, fontsize=fontsize, fontweight='bold', color=text_color,
                ha='center', va='center', zorder=4)

    def draw_rzz_gate(x, y1, y2, label, color=c_cost_pair):
        ax.plot([x, x], [y1, y2], color=color, lw=2.2, zorder=2)
        ax.plot(x, y1, 'o', color=color, markersize=6.5, markeredgecolor='#111827', markeredgewidth=1, zorder=3)
        ax.plot(x, y2, 'o', color=color, markersize=6.5, markeredgecolor='#111827', markeredgewidth=1, zorder=3)
        draw_box_gate(x, (y1 + y2) / 2, label, color=color, width=1.1, height=0.42, fontsize=7.5)

    def draw_two_qubit_gate(x, y1, y2, label, color):
        ax.plot([x, x], [y1, y2], color=color, lw=2.2, zorder=2)
        draw_box_gate(x, y1, label, color=color, width=0.85, height=0.48, fontsize=7.5)
        draw_box_gate(x, y2, label, color=color, width=0.85, height=0.48, fontsize=7.5)

    def draw_barrier(x, label_text):
        ax.axvline(x=x, color=c_barrier, linestyle='--', lw=1.3, zorder=2)
        bbox_props = dict(boxstyle="round,pad=0.3", fc="#F3F4F6", ec=c_barrier, lw=1)
        ax.text(x, n_qubits - 0.2, label_text, fontsize=9, fontweight='bold', color='#374151',
                ha='center', va='bottom', bbox=bbox_props, zorder=5)

    # 1. Quantum Wires
    for i in range(n_qubits):
        y = wire_y[i]
        ax.plot([x_wire_start, x_wire_end], [y, y], color=c_wire, lw=1.5, zorder=1)
        ax.text(x_wire_start - 0.25, y, f"$|0\\rangle_{{{i}}}$", fontsize=11, fontweight='bold',
                color='#111827', va='center', ha='right')

    # Stage 1: Warm-Start Prep
    for i in range(n_qubits):
        draw_box_gate(x_prep, wire_y[i], f"$R_y(\\theta_{{{i}}})$", color=c_prep, width=0.95)
    
    draw_barrier(x_barrier1, "1. Warm-Start Prep")

    # Stage 2: Cost Unitary U(C, gamma)
    for i in range(n_qubits):
        draw_box_gate(x_cost_single, wire_y[i], f"$R_z(2\\gamma Q_{{{i}{i}}})$", color=c_cost_single, width=1.2)

    for idx, (q1, q2) in enumerate(rzz_pairs):
        draw_rzz_gate(x_rzz_start + idx * 1.25, wire_y[q1], wire_y[q2], f"$R_{{zz}}(\\gamma \\tilde{{Q}}_{{{q1}{q2}}})$")
    
    draw_barrier(x_barrier2, "2. Cost Unitary $U(C, \\gamma)$")

    # Stage 3: Mixer Phase U(B, beta)
    if not xy_mixer:
        for i in range(n_qubits):
            draw_box_gate(x_mixer_start, wire_y[i], f"$R_y(-\\theta_{{{i}}})$", color=c_mixer, width=0.95)
            draw_box_gate(x_mixer_start + 1.1, wire_y[i], "$R_x(2\\beta)$", color=c_mixer, width=0.9)
            draw_box_gate(x_mixer_start + 2.2, wire_y[i], f"$R_y(\\theta_{{{i}}})$", color=c_mixer, width=0.95)
    else:
        for i in range(n_qubits - 1):
            draw_two_qubit_gate(x_mixer_start + i * 1.1, wire_y[i], wire_y[i+1], "$R_{xx}(\\beta)$", color=c_mixer)
        x_yy_start = x_mixer_start + (n_qubits - 1) * 1.1 + 0.2
        for i in range(n_qubits - 1):
            draw_two_qubit_gate(x_yy_start + i * 1.1, wire_y[i], wire_y[i+1], "$R_{yy}(\\beta)$", color='#991B1B')

    draw_barrier(x_barrier3, mixer_title)

    # Stage 4: Readout / Measurement
    for i in range(n_qubits):
        y = wire_y[i]
        draw_box_gate(x_meas, y, "M", color=c_meas, width=0.55, height=0.48)
        ax.plot([x_meas + 0.28, x_wire_end], [y - 0.04, y - 0.04], color='#4B5563', lw=1.0)
        ax.plot([x_meas + 0.28, x_wire_end], [y + 0.04, y + 0.04], color='#4B5563', lw=1.0)
        ax.text(x_wire_end + 0.15, y, f"$c_{{{i}}}$", fontsize=11, fontweight='bold', color='#111827', va='center')

    mixer_type = "Warm-Start X-Mixer" if not xy_mixer else "XY-Mixer"
    plt.title(f"WS-LR QAOA Quantum Circuit Architecture ($N = {n_qubits}$ Qubits, {mixer_type})",
              fontsize=12, fontweight='bold', pad=25, color='#111827')
    
    plt.xlim(0, x_wire_end + 0.6)
    plt.ylim(-0.6, n_qubits + 0.5)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

# Generate publication figures for N = 2, 3, 4 qubits
if __name__ == "__main__":
    for n in [2, 3, 4]:
        generate_publication_qaoa_circuit(n, xy_mixer=False, save_path=f"qaoa_pub_N{n}_x_mixer.pdf")
        generate_publication_qaoa_circuit(n, xy_mixer=True, save_path=f"qaoa_pub_N{n}_xy_mixer.pdf")