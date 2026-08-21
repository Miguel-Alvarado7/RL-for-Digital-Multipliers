`timescale 1ns/1ps

module testbench;

    reg [5:0] A;
    reg [5:0] B;
    wire [11:0] P;

    multiplier dut (
        .A(A),
        .B(B),
        .P(P)
    );

    initial begin
        // Caso 0: Prueba aleatoria 0
                A = 8'd54;
                B = 8'd22;
                #10;
                $display("%d", P);
// Caso 1: Prueba aleatoria 1
                A = 8'd22;
                B = 8'd41;
                #10;
                $display("%d", P);
// Caso 2: Prueba aleatoria 2
                A = 8'd11;
                B = 8'd28;
                #10;
                $display("%d", P);
// Caso 3: Prueba aleatoria 3
                A = 8'd59;
                B = 8'd8;
                #10;
                $display("%d", P);
// Caso 4: Prueba aleatoria 4
                A = 8'd32;
                B = 8'd62;
                #10;
                $display("%d", P);
// Caso 5: Prueba aleatoria 5
                A = 8'd36;
                B = 8'd55;
                #10;
                $display("%d", P);
// Caso 6: Prueba aleatoria 6
                A = 8'd52;
                B = 8'd35;
                #10;
                $display("%d", P);
// Caso 7: Prueba aleatoria 7
                A = 8'd47;
                B = 8'd43;
                #10;
                $display("%d", P);
        $finish;
    end

endmodule
