`timescale 1ns/1ps

// Testbench UNIVERSAL para los multiplicadores generados por el proyecto.
//
// Todos los DUT comparten la misma interfaz y solo cambia el ancho:
//     module multiplier (input [BITS-1:0] A, input [BITS-1:0] B,
//                        output [2*BITS-1:0] P);
// asi que UN solo testbench cubre cualquier --bits: el ancho se fija al
// compilar con -D BITS=<n> (por defecto 2) y el barrido es exhaustivo sobre
// todos los pares (A, B) en [0, 2^BITS).
//
// Uso:
//   iverilog -g2005 -D BITS=4 -o /tmp/sim \
//       out/montecarlo_cuda/bits4/best_multiplier_cuda.v \
//       verification/tb_multiplier_universal.v
//   vvp /tmp/sim
//
// Ademas del PASS/FAIL booleano imprime las metricas de error tipicas de
// computacion aproximada (MAE, peak error, % de respuestas exactas),
// acumuladas dentro del mismo barrido. El numero de productos parciales
// ("wires") no es visible desde el testbench: se obtiene fuera con
//     grep -c "wire pp" <dut>.v
//
// Rango practico: BITS entre 1 y ~12 (el barrido crece como 4^BITS; a 8
// bits son 65536 casos, ~1 s en vvp).

module tb_multiplier;
`ifndef BITS
`define BITS 2
`endif

    localparam integer BITS = `BITS;

    reg  [BITS-1:0]     A, B;
    wire [2*BITS-1:0]   P;

    integer a, b, ab, err;
    integer errors, exact, maxv;

    // Acumuladores a 64 bits: n_cases llega a 2^(2*BITS) y sum_err puede
    // superar los 32 bits ya con BITS=8 (peor caso ~2^16 x 2^16).
    reg [63:0] sum_err;
    reg [63:0] peak;
    reg [63:0] n_cases;

    real mae, pct_exact;

    // Detalle de fallos impreso: con circuitos aproximados puede haber miles
    // de mismatches y volcarlos todos inunda la salida.
    localparam MAX_FAIL_LINES = 20;

    multiplier dut (.A(A), .B(B), .P(P));

    initial begin
        errors  = 0;
        exact   = 0;
        sum_err = 0;
        peak    = 0;
        maxv    = 1 << BITS;
        n_cases = maxv;
        n_cases = n_cases * maxv;           // dos pasos para no desbordar 32 bits

        $display("=== Test de multiplier (%0d-bit, %0d casos) ===", BITS, n_cases);

        for (a = 0; a < maxv; a = a + 1) begin
            for (b = 0; b < maxv; b = b + 1) begin
                A = a;                      // truncamiento implicito a [BITS-1:0]
                B = b;
                #10;
                ab = A * B;                 // cero-extension a 32 bits: sin overflow
                if (P !== ab) begin
                    err = (P > ab) ? (P - ab) : (ab - P);
                    errors = errors + 1;
                    if (errors <= MAX_FAIL_LINES)
                        $display("FAIL: A=%0d B=%0d -> P=%0d (esperado %0d)",
                                 A, B, P, ab);
                    else if (errors == MAX_FAIL_LINES + 1)
                        $display("... (se omiten el resto de fallos)");
                end else begin
                    err = 0;
                    exact = exact + 1;
                end
                sum_err = sum_err + err;
                if (err > peak)
                    peak = err;
            end
        end

        mae       = sum_err / (n_cases * 1.0);   // division real, no entera
        pct_exact = 100.0 * exact / n_cases;

        if (errors == 0)
            $display("TEST PASSED (%0d/%0d)", n_cases, n_cases);
        else
            $display("TEST FAILED (%0d errores de %0d)", errors, n_cases);
        $display("MAE: %.4f", mae);
        $display("Peak Error: %0d", peak);
        $display("Respuestas Exactas: %.2f%% (%0d/%0d)",
                 pct_exact, exact, n_cases);
        $finish;
    end

endmodule
