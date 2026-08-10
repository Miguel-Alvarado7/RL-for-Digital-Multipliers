`timescale 1ns/1ps

module test_multiplier_4bit_tb;

    reg [3:0] A;
    reg [3:0] B;
    wire [7:0] P;

    integer a, b, errors;

    multiplier dut (
        .A(A),
        .B(B),
        .P(P)
    );

    initial begin
        errors = 0;
        $display("=== Test de multiplier (4-bit) ===");
        for (a = 0; a < 16; a = a + 1) begin
            for (b = 0; b < 16; b = b + 1) begin
                A = a[3:0];
                B = b[3:0];
                #10;
                if (P !== (A * B)) begin
                    $display("FAIL: A=%d B=%d -> P=%d (esperado %d)", A, B, P, A * B);
                    errors = errors + 1;
                end else begin
                    $display("PASS: A=%d B=%d -> P=%d", A, B, P);
                end
            end
        end
        if (errors == 0)
            $display("TEST PASSED (256/256)");
        else
            $display("TEST FAILED (%0d errores)", errors);
        $finish;
    end

endmodule