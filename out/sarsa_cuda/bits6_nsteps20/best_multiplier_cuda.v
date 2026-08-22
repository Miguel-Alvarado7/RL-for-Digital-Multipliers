
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
    wire pp5 = (A[4]&B[3]);
    wire pp6 = (A[3]&B[0]);
    wire pp7 = (A[5]&B[1]);
    wire pp8 = (A[1]&B[1]);
    wire pp9 = (A[0]&~B[3]);
    wire pp10 = 0;
    wire pp11 = (A[3]&B[5]);
    wire pp12 = (A[4]&~B[4]);
    wire pp13 = (~A[5]&~B[4]);
    wire pp14 = (A[1]&B[2]);
    wire pp15 = (~A[4]&B[3]);
    wire pp16 = (A[5]&~B[4]);
    wire pp17 = (A[2]&~B[3]);
    wire pp18 = (A[3]&B[3]);
    wire pp19 = (A[2]&B[5]);
    wire pp20 = (A[0]&B[1]);
    wire pp21 = (A[5]&B[2]);
    wire pp22 = (~A[1]&B[0]);
    wire pp23 = (A[4]&~B[1]);
    wire pp24 = (~A[2]&B[1]);
    wire pp25 = (A[2]&~B[0]);
    wire pp26 = (~A[3]&B[2]);
    wire pp27 = (A[2]&B[4]);
    wire pp28 = (~A[3]&B[3]);
    wire pp29 = (A[5]&~B[0]);
    wire pp30 = (A[2]&B[2]);
    wire pp31 = (~A[0]&B[3]);
    wire pp32 = (A[4]&B[2]);
    wire pp33 = (~A[0]&~B[1]);
    wire pp34 = (A[0]&B[2]);

    // Suma de productos parciales
    wire [5:0] columna12 = pp0 + pp10;
    wire [5:0] columna11 = pp1 + pp10;
    wire [5:0] columna10 = pp2 + pp10;
    wire [5:0] columna9 = pp3 + pp10;
    wire [5:0] columna8 = pp4 + pp11 + pp10;
    wire [5:0] columna7 = pp5 + pp12 + pp18 + pp10;
    wire [5:0] columna6 = pp6 + pp13 + pp19 + pp18 + pp10;
    wire [5:0] columna5 = pp7 + pp14 + pp20 + pp25 + pp30 + pp10;
    wire [5:0] columna4 = pp2 + pp15 + pp21 + pp26 + pp31 + pp10;
    wire [5:0] columna3 = pp2 + pp16 + pp22 + pp27 + pp32 + pp10;
    wire [5:0] columna2 = pp8 + pp0 + pp23 + pp28 + pp33 + pp10;
    wire [5:0] columna1 = pp9 + pp17 + pp24 + pp29 + pp34 + pp10;
    assign P = (columna12 << 11) + (columna11 << 10) + (columna10 << 9) + (columna9 << 8) + (columna8 << 7) + (columna7 << 6) + (columna6 << 5) + (columna5 << 4) + (columna4 << 3) + (columna3 << 2) + (columna2 << 1) + (columna1 << 0);

endmodule