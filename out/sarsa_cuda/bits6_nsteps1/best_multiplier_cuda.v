
`timescale 1ns/1ps
module multiplier (
    input [5:0] A,
    input [5:0] B,
    output [11:0] P);

    // Generación de productos parciales (sin redundancias)
    wire pp0 = (A[5]&B[5]);
    wire pp1 = (A[4]&B[4]);
    wire pp2 = 0;
    wire pp3 = (~A[5]&B[5]);
    wire pp4 = (A[5]&~B[5]);
    wire pp5 = (A[4]&~B[4]);
    wire pp6 = (~A[0]&B[1]);
    wire pp7 = (~A[4]&~B[5]);
    wire pp8 = (A[2]&B[2]);
    wire pp9 = (A[3]&~B[3]);
    wire pp10 = (~A[1]&~B[3]);
    wire pp11 = (A[3]&B[2]);
    wire pp12 = (~A[4]&B[4]);
    wire pp13 = (~A[3]&B[3]);
    wire pp14 = (A[2]&~B[0]);
    wire pp15 = (~A[1]&~B[0]);
    wire pp16 = (A[5]&B[4]);
    wire pp17 = (A[4]&~B[5]);
    wire pp18 = (A[4]&B[3]);
    wire pp19 = (~A[0]&B[4]);
    wire pp20 = (A[3]&B[3]);
    wire pp21 = (~A[0]&~B[4]);

    // Suma de productos parciales
    wire [5:0] columna12 = pp0 + pp2;
    wire [5:0] columna11 = pp1 + pp2;
    wire [5:0] columna10 = pp2;
    wire [5:0] columna9 = pp3 + pp2 + pp20;
    wire [5:0] columna8 = pp4 + pp12 + pp2;
    wire [5:0] columna7 = pp5 + pp2;
    wire [5:0] columna6 = pp6 + pp9 + pp2;
    wire [5:0] columna5 = pp7 + pp2;
    wire [5:0] columna4 = pp8 + pp13 + pp2;
    wire [5:0] columna3 = pp9 + pp12 + pp2;
    wire [5:0] columna2 = pp10 + pp4 + pp8 + pp16 + pp18 + pp2;
    wire [5:0] columna1 = pp11 + pp14 + pp15 + pp17 + pp19 + pp21;
    assign P = (columna12 << 11) + (columna11 << 10) + (columna10 << 9) + (columna9 << 8) + (columna8 << 7) + (columna7 << 6) + (columna6 << 5) + (columna5 << 4) + (columna4 << 3) + (columna3 << 2) + (columna2 << 1) + (columna1 << 0);

endmodule