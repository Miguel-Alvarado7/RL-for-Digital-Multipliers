
`timescale 1ns/1ps
module multiplier (
    input [5:0] A,
    input [5:0] B,
    output [11:0] P);

    // Generación de productos parciales (sin redundancias)
    wire pp0 = 0;
    wire pp1 = (A[5]&~B[3]);
    wire pp2 = (A[4]&B[3]);
    wire pp3 = (~A[2]&B[0]);
    wire pp4 = (~A[1]&~B[0]);
    wire pp5 = (A[5]&~B[2]);
    wire pp6 = (A[0]&~B[3]);
    wire pp7 = (~A[2]&B[3]);
    wire pp8 = (~A[5]&~B[4]);
    wire pp9 = (A[0]&B[4]);
    wire pp10 = (~A[1]&B[0]);
    wire pp11 = (~A[2]&~B[5]);
    wire pp12 = (~A[5]&~B[3]);
    wire pp13 = (A[5]&B[4]);
    wire pp14 = (~A[2]&B[4]);
    wire pp15 = (A[2]&B[1]);
    wire pp16 = (A[1]&B[2]);
    wire pp17 = (~A[5]&~B[0]);
    wire pp18 = (~A[4]&~B[2]);
    wire pp19 = (A[3]&B[5]);
    wire pp20 = (~A[4]&~B[3]);
    wire pp21 = (~A[4]&B[4]);
    wire pp22 = (A[4]&~B[2]);
    wire pp23 = (~A[4]&B[5]);
    wire pp24 = (A[2]&~B[4]);
    wire pp25 = (A[0]&B[2]);
    wire pp26 = (A[5]&B[1]);
    wire pp27 = (~A[2]&B[1]);
    wire pp28 = (A[3]&~B[3]);
    wire pp29 = (~A[5]&B[3]);
    wire pp30 = (A[5]&B[5]);
    wire pp31 = (A[3]&B[3]);
    wire pp32 = (~A[3]&B[0]);
    wire pp33 = (~A[0]&~B[5]);
    wire pp34 = (A[3]&B[4]);
    wire pp35 = (~A[2]&B[2]);
    wire pp36 = (A[0]&~B[1]);
    wire pp37 = (A[3]&~B[5]);
    wire pp38 = (A[5]&B[2]);
    wire pp39 = (A[2]&B[4]);
    wire pp40 = (~A[4]&B[0]);
    wire pp41 = (A[4]&~B[5]);
    wire pp42 = (~A[4]&B[3]);
    wire pp43 = (~A[3]&~B[1]);
    wire pp44 = (A[0]&~B[0]);
    wire pp45 = (A[5]&B[3]);
    wire pp46 = (~A[0]&B[3]);
    wire pp47 = (A[5]&~B[0]);
    wire pp48 = (A[4]&B[4]);
    wire pp49 = (~A[5]&B[4]);

    // Suma de productos parciales
    wire [5:0] columna12 = pp0 + pp11 + pp20;
    wire [5:0] columna11 = pp1 + pp11 + pp20;
    wire [5:0] columna10 = pp2 + pp11 + pp19 + pp30 + pp20 + pp45;
    wire [5:0] columna9 = pp3 + pp12 + pp21 + pp20 + pp8 + pp11;
    wire [5:0] columna8 = pp4 + pp13 + pp22 + pp31 + pp37 + pp15;
    wire [5:0] columna7 = pp5 + pp14 + pp23 + pp32 + pp38 + pp46;
    wire [5:0] columna6 = pp6 + pp15 + pp24 + pp12 + pp39;
    wire [5:0] columna5 = pp7 + pp16 + pp25 + pp33 + pp40 + pp9;
    wire [5:0] columna4 = pp0 + pp17 + pp26 + pp34 + pp41 + pp47;
    wire [5:0] columna3 = pp8 + pp18 + pp27 + pp20 + pp42 + pp48;
    wire [5:0] columna2 = pp9 + pp6 + pp28 + pp35 + pp43 + pp31;
    wire [5:0] columna1 = pp10 + pp19 + pp29 + pp36 + pp44 + pp49;
    assign P = (columna12 << 11) + (columna11 << 10) + (columna10 << 9) + (columna9 << 8) + (columna8 << 7) + (columna7 << 6) + (columna6 << 5) + (columna5 << 4) + (columna4 << 3) + (columna3 << 2) + (columna2 << 1) + (columna1 << 0);

endmodule