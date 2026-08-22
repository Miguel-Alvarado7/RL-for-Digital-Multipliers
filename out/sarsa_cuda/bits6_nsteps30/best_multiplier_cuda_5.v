
`timescale 1ns/1ps
module multiplier (
    input [5:0] A,
    input [5:0] B,
    output [11:0] P);

    // Generación de productos parciales (sin redundancias)
    wire pp0 = (A[5]&B[5]);
    wire pp1 = (A[4]&B[4]);
    wire pp2 = (A[5]&~B[5]);
    wire pp3 = (~A[5]&B[5]);
    wire pp4 = (~A[4]&B[4]);
    wire pp5 = (A[3]&B[2]);
    wire pp6 = (A[2]&B[3]);
    wire pp7 = (~A[0]&B[3]);
    wire pp8 = (~A[1]&~B[2]);
    wire pp9 = (~A[2]&~B[5]);
    wire pp10 = (A[3]&~B[3]);
    wire pp11 = (A[4]&~B[0]);
    wire pp12 = 0;
    wire pp13 = (A[3]&B[5]);
    wire pp14 = (A[3]&B[3]);
    wire pp15 = (A[2]&B[4]);
    wire pp16 = (A[4]&B[3]);
    wire pp17 = (~A[1]&B[2]);
    wire pp18 = (~A[4]&~B[5]);
    wire pp19 = (A[1]&B[4]);
    wire pp20 = (~A[4]&~B[1]);
    wire pp21 = (~A[5]&B[2]);
    wire pp22 = (~A[2]&B[2]);
    wire pp23 = (~A[3]&B[3]);
    wire pp24 = (~A[3]&B[4]);
    wire pp25 = (A[4]&B[2]);
    wire pp26 = (~A[1]&B[1]);
    wire pp27 = (A[5]&B[2]);
    wire pp28 = (A[4]&B[1]);
    wire pp29 = (~A[5]&~B[4]);
    wire pp30 = (A[4]&~B[4]);
    wire pp31 = (A[1]&B[5]);
    wire pp32 = (A[2]&~B[0]);

    // Suma de productos parciales
    wire [5:0] columna12 = pp0 + pp12;
    wire [5:0] columna11 = pp1 + pp12;
    wire [5:0] columna10 = pp2 + pp12;
    wire [5:0] columna9 = pp3 + pp12;
    wire [5:0] columna8 = pp4 + pp13 + pp12;
    wire [5:0] columna7 = pp5 + pp14 + pp12 + pp30;
    wire [5:0] columna6 = pp6 + pp15 + pp13 + pp25 + pp29 + pp12;
    wire [5:0] columna5 = pp7 + pp16 + pp20 + pp26 + pp12;
    wire [5:0] columna4 = pp8 + pp17 + pp21 + pp27 + pp31 + pp12;
    wire [5:0] columna3 = pp9 + pp18 + pp22 + pp28 + pp32 + pp12;
    wire [5:0] columna2 = pp10 + pp19 + pp23 + pp29 + pp21 + pp12;
    wire [5:0] columna1 = pp11 + pp12 + pp24 + pp1 + pp14;
    assign P = (columna12 << 11) + (columna11 << 10) + (columna10 << 9) + (columna9 << 8) + (columna8 << 7) + (columna7 << 6) + (columna6 << 5) + (columna5 << 4) + (columna4 << 3) + (columna3 << 2) + (columna2 << 1) + (columna1 << 0);

endmodule