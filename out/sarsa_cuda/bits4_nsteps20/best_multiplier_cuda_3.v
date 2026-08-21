
`timescale 1ns/1ps
module multiplier (
    input [3:0] A,
    input [3:0] B,
    output [7:0] P);

    // Generación de productos parciales (sin redundancias)
    wire pp0 = 0;
    wire pp1 = (A[3]&B[3]);
    wire pp2 = (A[3]&B[2]);
    wire pp3 = (A[2]&B[2]);
    wire pp4 = (A[2]&B[1]);
    wire pp5 = (A[0]&B[2]);
    wire pp6 = (A[1]&B[2]);
    wire pp7 = (A[0]&B[0]);
    wire pp8 = (A[2]&B[3]);
    wire pp9 = (A[3]&B[1]);
    wire pp10 = (A[3]&B[0]);
    wire pp11 = (A[1]&B[1]);
    wire pp12 = (A[1]&B[3]);
    wire pp13 = (A[0]&B[3]);
    wire pp14 = (A[2]&B[0]);

    // Suma de productos parciales
    wire [3:0] columna8 = pp0;
    wire [3:0] columna7 = pp1 + pp0;
    wire [3:0] columna6 = pp2 + pp8 + pp0;
    wire [3:0] columna5 = pp3 + pp9 + pp12;
    wire [3:0] columna4 = pp4 + pp10 + pp13;
    wire [3:0] columna3 = pp5 + pp11 + pp6;
    wire [3:0] columna2 = pp6 + pp10 + pp14;
    wire [3:0] columna1 = pp7 + pp4 + pp14;
    assign P = (columna8 << 7) + (columna7 << 6) + (columna6 << 5) + (columna5 << 4) + (columna4 << 3) + (columna3 << 2) + (columna2 << 1) + (columna1 << 0);

endmodule