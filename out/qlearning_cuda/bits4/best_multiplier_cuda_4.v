
`timescale 1ns/1ps
module multiplier (
    input [3:0] A,
    input [3:0] B,
    output [7:0] P);

    // Generación de productos parciales (sin redundancias)
    wire pp0 = (A[3]&B[3]);
    wire pp1 = 0;
    wire pp2 = (A[1]&B[1]);
    wire pp3 = (~A[3]&B[3]);
    wire pp4 = (A[3]&~B[3]);
    wire pp5 = (A[2]&B[0]);
    wire pp6 = (A[2]&~B[2]);
    wire pp7 = (A[2]&B[2]);
    wire pp8 = (A[0]&B[0]);
    wire pp9 = (~A[3]&B[2]);
    wire pp10 = (A[0]&B[2]);

    // Suma de productos parciales
    wire [3:0] columna8 = pp0 + pp1;
    wire [3:0] columna7 = pp1;
    wire [3:0] columna6 = pp2 + pp7 + pp1;
    wire [3:0] columna5 = pp3 + pp1;
    wire [3:0] columna4 = pp4 + pp8 + pp1;
    wire [3:0] columna3 = pp5 + pp1;
    wire [3:0] columna2 = pp4 + pp1 + pp9;
    wire [3:0] columna1 = pp6 + pp1 + pp10;
    assign P = (columna8 << 7) + (columna7 << 6) + (columna6 << 5) + (columna5 << 4) + (columna4 << 3) + (columna3 << 2) + (columna2 << 1) + (columna1 << 0);

endmodule