Sub ChangeFontSizeAndBold()
    Dim ws As Worksheet
    Dim chartObj As ChartObject
    Dim chart As chart
    Dim fontSize As Integer
    Dim series As series

    ' フォントサイズを指定
    fontSize = 16

    ' 現在のシートを対象にする
    Set ws = ActiveSheet

    ' シート上の全てのグラフオブジェクトに対して処理
    For Each chartObj In ws.ChartObjects
        Set chart = chartObj.chart

        ' 図形（グラフオブジェクト）の枠線をなしに設定
        chartObj.ShapeRange.Line.Visible = msoFalse

        ' プロットエリアの枠線を黒、幅を1.25に設定
        chart.PlotArea.Format.Line.ForeColor.RGB = RGB(0, 0, 0)
        chart.PlotArea.Format.Line.Weight = 1.25

        ' 縦軸の線を黒、幅を1.25に設定
        chart.Axes(xlValue).Format.Line.ForeColor.RGB = RGB(0, 0, 0)
        chart.Axes(xlValue).Format.Line.Weight = 1.25

        ' 横軸の線を黒、幅を1.25に設定
        chart.Axes(xlCategory).Format.Line.ForeColor.RGB = RGB(0, 0, 0)
        chart.Axes(xlCategory).Format.Line.Weight = 1.25

        ' 縦目盛り線を黒、幅を0.5に設定
        chart.Axes(xlValue).MajorGridlines.Format.Line.ForeColor.RGB = RGB(160, 160, 160)
        chart.Axes(xlValue).MajorGridlines.Format.Line.Weight = 0.5

        ' 縦補助目盛り線を黒、幅を0.5に設定
        chart.Axes(xlValue).MinorGridlines.Format.Line.ForeColor.RGB = RGB(225, 225, 225)
        chart.Axes(xlValue).MinorGridlines.Format.Line.Weight = 0.5

        ' 横目盛り線を黒、幅を0.5に設定
        chart.Axes(xlCategory).MajorGridlines.Format.Line.ForeColor.RGB = RGB(160, 160, 160)
        chart.Axes(xlCategory).MajorGridlines.Format.Line.Weight = 0.5

        ' 縦補助盛り線を黒、幅を0.5に設定
        chart.Axes(xlCategory).MinorGridlines.Format.Line.ForeColor.RGB = RGB(225, 225, 225)
        chart.Axes(xlCategory).MinorGridlines.Format.Line.Weight = 0.5

        ' X軸ラベルのフォント設定
        chart.Axes(xlCategory).TickLabels.Font.Size = fontSize
        chart.Axes(xlCategory).TickLabels.Font.Bold = False
        chart.Axes(xlCategory).TickLabels.Font.Color = RGB(0, 0, 0)

        ' Y軸ラベルのフォント設定
        chart.Axes(xlValue).TickLabels.Font.Size = fontSize
        chart.Axes(xlValue).TickLabels.Font.Bold = False
        chart.Axes(xlValue).TickLabels.Font.Color = RGB(0, 0, 0)

        ' X軸タイトルの設定
        If Not chart.Axes(xlCategory, xlPrimary).HasTitle Then
            chart.Axes(xlCategory, xlPrimary).HasTitle = True
        End If
        chart.Axes(xlCategory, xlPrimary).AxisTitle.Text = "Frequency (Hz)"
        chart.Axes(xlCategory, xlPrimary).AxisTitle.Font.Size = fontSize + 2
        chart.Axes(xlCategory, xlPrimary).AxisTitle.Font.Bold = False
        chart.Axes(xlCategory, xlPrimary).AxisTitle.Font.Color = RGB(0, 0, 0)
        '=== X軸（横軸）を対数（基数10）にし、範囲と目盛りを設定（安全版） ===
        ' ライン/縦棒などの「カテゴリ軸」では MajorUnit は設定不可のため、
        ' XY散布図など「値軸」の時のみ MajorUnit を設定し、その他はエラーを無視してグリッド等だけセットする
        Dim xAx As Axis
        On Error Resume Next
        Set xAx = chart.Axes(xlCategory, xlPrimary)
        On Error GoTo 0
        If Not xAx Is Nothing Then
            Call SetupLogXAxis(xAx, 10#, 20000#)
        End If

        ' Y軸タイトルの設定
        If Not chart.Axes(xlValue, xlPrimary).HasTitle Then
            chart.Axes(xlValue, xlPrimary).HasTitle = True
        End If
        chart.Axes(xlValue, xlPrimary).AxisTitle.Text = "Gain (dB)"
        chart.Axes(xlValue, xlPrimary).AxisTitle.Font.Size = fontSize + 2
        chart.Axes(xlValue, xlPrimary).AxisTitle.Font.Bold = False
        chart.Axes(xlValue, xlPrimary).AxisTitle.Font.Color = RGB(0, 0, 0)

        '=== Y軸（縦軸）のスケーリング設定 ===
        ' 主目盛り：10 dB刻み、補助目盛り：5 dB刻み
        ' 最小値：-40 dB、最大値：プロットされている最大値の1の位を切り上げ
        Dim yMax As Double, yMaxCeil As Double
        yMax = GetChartMaxY(chart)
        yMaxCeil = WorksheetFunction.Ceiling(yMax, 10)
        With chart.Axes(xlValue)
            .MinimumScaleIsAuto = False
            .MaximumScaleIsAuto = False
            .MinimumScale = -40
            .MaximumScale = yMaxCeil
            .MajorUnit = 10
            .MinorUnit = 5
            .HasMinorGridlines = True
            .MajorTickMark = xlOutside
            .MinorTickMark = xlOutside
        End With

        ' 凡例のフォントサイズ、太字解除、フォントカラーを黒に設定
        If chart.HasLegend Then
            With chart.Legend
                ' フォントの設定
                With .Font
                    .Size = fontSize - 2
                    .Bold = False
                    .Color = RGB(0, 0, 0)
                End With

                ' 塗りつぶしの設定
                With .Format.Fill
                    .ForeColor.RGB = RGB(255, 255, 255) ' 白で塗りつぶし
                    .Transparency = 0.1 ' 透明度10%
                End With

                ' 凡例のサイズと位置を設定（ここを追加）
                 '.Left = 100 ' 左端からの位置（単位: ポイント）
                '.Top = 30 ' 上端からの位置（単位: ポイント）
                '.Width = 100 ' 幅（単位: ポイント）
                '.Height = 100 ' 高さ（単位: ポイント）
            End With
        End If

        '--- 線色と透明度をシリーズ番号に応じて設定（1→11で不透明度UP）
    Const BASE_R As Long = 10
    Const BASE_G As Long = 62
    Const BASE_B As Long = 85

    Const MAX_SERIES As Long = 11        ' 1?11で段階付け
    Const TRANS_START As Double = 0.8    ' Series 1 の透明度（0?1）
    Const TRANS_END   As Double = 0.05    ' Series 11 の透明度（0?1）

    Dim s As series
    Dim idx As Long
    Dim frac As Double
    Dim t As Double

    idx = 1
    For Each s In chart.SeriesCollection
        With s.Format.Line
            .Visible = msoTrue
            .Weight = 1.75
            .ForeColor.RGB = RGB(BASE_R, BASE_G, BASE_B)

            ' 1→11で transparency を直線的に TRANS_START → TRANS_END へ
            frac = WorksheetFunction.Min(idx - 1, MAX_SERIES - 1) / (MAX_SERIES - 1)
            t = TRANS_START + (TRANS_END - TRANS_START) * frac
            If t < 0 Then t = 0
            If t > 1 Then t = 1
            .Transparency = t
        End With
        idx = idx + 1
    Next s

        ' データラベルのフォントサイズ、太字解除、フォントカラーを黒に設定
        For Each series In chart.SeriesCollection
            If series.HasDataLabels Then
                series.DataLabels.Font.Size = fontSize
                series.DataLabels.Font.Bold = False
                series.DataLabels.Font.Color = RGB(0, 0, 0)
            End If
        Next series

        '=== グラフタイトルを「下」に配置（標準のChartTitleは上なので非表示にしてTextBoxで再配置） ===
        Dim titleText As String
        titleText = ""
        If chart.HasTitle Then
            titleText = chart.ChartTitle.Text
        End If
        ' 上部の標準タイトルは使わない
        chart.HasTitle = False
        ' 下配置タイトルをセット（titleText が空のときは既存の下部タイトルを優先して引き継ぐ）
        Call SetChartBottomTitle(chart, titleText, fontSize + 8)
    Next chartObj
End Sub

Private Sub SetupLogXAxis(xAx As Axis, minVal As Double, maxVal As Double)
    ' X軸を対数(基数10)に設定。値軸の場合のみ MajorUnit=1 を適用。
    ' カテゴリ軸では MajorUnit は設定不可のため、エラーを握りつつ補助目盛等のみ反映する。
    On Error Resume Next
    With xAx
        .ScaleType = xlLogarithmic
        .LogBase = 10
        .MinimumScaleIsAuto = False
        .MaximumScaleIsAuto = False
        .MinimumScale = minVal
        .MaximumScale = maxVal

        ' 値軸（xlValue）だけ MajorUnit を設定可能。カテゴリ軸ではエラーになるため握る。
        .MajorUnitIsAuto = False
        Err.Clear
        .MajorUnit = 1     ' 1デケード刻み（10^n）
        If Err.Number <> 0 Then
            ' カテゴリ軸など：MajorUnit 未対応 → 何もしない
            Err.Clear
        End If

        ' 補助目盛は Excel の対数軸既定（2?9倍）を利用
        .MinorUnitIsAuto = True
        .HasMinorGridlines = True
        .MinorTickMark = xlOutside
        .MajorTickMark = xlOutside

        ' ラベルの書式は参照セルに追従
        .TickLabels.NumberFormatLinked = True
    End With
    On Error GoTo 0
End Sub

Private Sub SetChartBottomTitle(ch As chart, titleText As String, fontSizePt As Integer)
    Dim shp As Shape
    Dim found As Boolean
    Dim existingText As String
    
    '=== タイトル用のスペースを下部に確保する ===
    Dim titleSpacePts As Single
    titleSpacePts = fontSizePt * 2.5 + 1  ' title box height + margin
    Call EnsureBottomPlotSpace(ch, titleSpacePts)
    
    ' 既存の下部タイトル（名前：BottomTitle）を探す
    found = False
    existingText = ""
    For Each shp In ch.Shapes
        If shp.Name = "BottomTitle" Then
            found = True
            On Error Resume Next
            existingText = shp.TextFrame2.TextRange.Text
            On Error GoTo 0
            Exit For
        End If
    Next shp
    
    ' 見つからなければ新規作成（横書きテキストボックス）
    If Not found Then
        If Len(Trim$(titleText)) = 0 Then
            titleText = ch.Name ' 最終フォールバック（既存も上部も無い場合）
        End If
        Set shp = ch.Shapes.AddTextbox( _
                    Orientation:=msoTextOrientationHorizontal, _
                    Left:=ch.PlotArea.InsideLeft, _
                    Top:=ch.PlotArea.InsideTop + ch.PlotArea.InsideHeight + 6, _
                    Width:=ch.PlotArea.InsideWidth, _
                    Height:=fontSizePt * 1.8)
        shp.Name = "BottomTitle"
    End If
    
    ' 位置とサイズを毎回更新（プロットエリアに追従させる）
    With shp
        .Left = ch.PlotArea.InsideLeft
        ' X軸ラベルの直下 30pt に配置
        .Top = ch.Axes(xlCategory).Top + ch.Axes(xlCategory).Height + 30
        .Width = ch.PlotArea.InsideWidth
        .Height = fontSizePt * 1.8
        .Line.Visible = msoFalse
        .Fill.Visible = msoFalse
        
        With .TextFrame2
            .AutoSize = msoAutoSizeNone
            .MarginTop = 0
            .MarginBottom = 0
            .MarginLeft = 0
            .MarginRight = 0
            .TextRange.ParagraphFormat.Alignment = msoAlignCenter
            .VerticalAnchor = msoAnchorMiddle
            ' 既存タイトルがあり、titleText が空ならテキストは変更しない
            If Not found Or Len(Trim$(titleText)) > 0 Then
                .TextRange.Characters.Text = titleText
            End If
            With .TextRange.Characters.Font
                .Name = "MSP Gothic"
                .Size = fontSizePt
                .Bold = msoFalse
                .Fill.ForeColor.RGB = RGB(0, 0, 0)
            End With
        End With
    End With
    
    ' 前面に
    shp.ZOrder msoBringToFront
End Sub

Private Sub EnsureBottomPlotSpace(ch As chart, spacePts As Single)
    On Error Resume Next
    ' （外側の）プロットエリア下部の利用可能スペースを計算
    Dim avail As Single
    Dim delta As Single

    avail = ch.ChartArea.Height - (ch.PlotArea.Top + ch.PlotArea.Height)

    If avail + 0.5 < spacePts Then
        delta = spacePts - avail
        ' 過度に縮小しないように：プロットの高さは最低60ptを確保
        If ch.PlotArea.Height > delta + 60 Then
            ch.PlotArea.Height = ch.PlotArea.Height - delta
        Else
            ' フォールバック：これ以上高さを縮められない場合はプロットを上に移動
            ch.PlotArea.Top = Application.Max(0, ch.PlotArea.Top - (delta - (ch.PlotArea.Height - 60)))
            ch.PlotArea.Height = 60
        End If
    End If
End Sub

Private Function GetChartMaxY(ch As chart) As Double
    Dim s As series
    Dim v As Variant
    Dim curMax As Double
    Dim tmp As Double
    curMax = -1E+308
    On Error Resume Next
    For Each s In ch.SeriesCollection
        v = s.Values
        If IsArray(v) Then
            tmp = Application.WorksheetFunction.Max(v)
        Else
            tmp = CDbl(v)
        End If
        If Err.Number = 0 Then
            If tmp > curMax Then curMax = tmp
        Else
            Err.Clear
        End If
    Next s
    On Error GoTo 0
    If curMax = -1E+308 Then
        curMax = 0  ' フォールバック
    End If
    GetChartMaxY = curMax
End Function

Private Function Ceil1(x As Double) As Double
    ' 1の位で切り上げ（正の値：次の整数へ、負の値：0に近づく方向ではなく数学的な天井）
    If x >= 0 Then
        If x = Int(x) Then
            Ceil1 = x
        Else
            Ceil1 = Int(x) + 1
        End If
    Else
        ' 数学的なceil：負の値では -Int(-x)
        Ceil1 = -Int(-x)
    End If
End Function

'========================
' グラフタイトルなし版
'========================
Sub ChangeFontSizeAndBold_NoTitle()
    Dim ws As Worksheet
    Dim chartObj As ChartObject
    Dim chart As chart
    Dim fontSize As Integer
    Dim series As series

    ' フォントサイズを指定
    fontSize = 16

    ' 現在のシートを対象にする
    Set ws = ActiveSheet

    ' シート上の全てのグラフオブジェクトに対して処理
    For Each chartObj In ws.ChartObjects
        Set chart = chartObj.chart

        ' 図形（グラフオブジェクト）の枠線をなしに設定
        chartObj.ShapeRange.Line.Visible = msoFalse

        ' プロットエリアの枠線を黒、幅を1.25に設定
        chart.PlotArea.Format.Line.ForeColor.RGB = RGB(120, 120, 120)
        chart.PlotArea.Format.Line.Weight = 1.25

        ' 縦軸の線を黒、幅を1.25に設定
        chart.Axes(xlValue).Format.Line.ForeColor.RGB = RGB(120, 120, 120)
        chart.Axes(xlValue).Format.Line.Weight = 1.25

        ' 横軸の線を黒、幅を1.25に設定
        chart.Axes(xlCategory).Format.Line.ForeColor.RGB = RGB(120, 120, 120)
        chart.Axes(xlCategory).Format.Line.Weight = 1.25

        ' 縦目盛り線（主）をグレー、幅0.5に設定
        chart.Axes(xlValue).MajorGridlines.Format.Line.ForeColor.RGB = RGB(160, 160, 160)
        chart.Axes(xlValue).MajorGridlines.Format.Line.Weight = 0.5

        ' 縦目盛り線（補助）を薄いグレー、幅0.5に設定
        chart.Axes(xlValue).MinorGridlines.Format.Line.ForeColor.RGB = RGB(225, 225, 225)
        chart.Axes(xlValue).MinorGridlines.Format.Line.Weight = 0.5

        ' 横目盛り線（主）をグレー、幅0.5に設定
        chart.Axes(xlCategory).MajorGridlines.Format.Line.ForeColor.RGB = RGB(160, 160, 160)
        chart.Axes(xlCategory).MajorGridlines.Format.Line.Weight = 0.5

        ' 横目盛り線（補助）を薄いグレー、幅0.5に設定
        chart.Axes(xlCategory).MinorGridlines.Format.Line.ForeColor.RGB = RGB(225, 225, 225)
        chart.Axes(xlCategory).MinorGridlines.Format.Line.Weight = 0.5

        ' X軸ラベルのフォント設定
        chart.Axes(xlCategory).TickLabels.Font.Size = fontSize
        chart.Axes(xlCategory).TickLabels.Font.Bold = False
        chart.Axes(xlCategory).TickLabels.Font.Color = RGB(0, 0, 0)

        ' Y軸ラベルのフォント設定
        chart.Axes(xlValue).TickLabels.Font.Size = fontSize
        chart.Axes(xlValue).TickLabels.Font.Bold = False
        chart.Axes(xlValue).TickLabels.Font.Color = RGB(0, 0, 0)

        ' X軸タイトルの設定
        If Not chart.Axes(xlCategory, xlPrimary).HasTitle Then
            chart.Axes(xlCategory, xlPrimary).HasTitle = True
        End If
        chart.Axes(xlCategory, xlPrimary).AxisTitle.Text = "Frequency (Hz)"
        chart.Axes(xlCategory, xlPrimary).AxisTitle.Font.Size = fontSize + 2
        chart.Axes(xlCategory, xlPrimary).AxisTitle.Font.Bold = False
        chart.Axes(xlCategory, xlPrimary).AxisTitle.Font.Color = RGB(0, 0, 0)

        '=== X軸（横軸）対数設定・範囲・目盛 ===
        Dim xAx As Axis
        On Error Resume Next
        Set xAx = chart.Axes(xlCategory, xlPrimary)
        On Error GoTo 0
        If Not xAx Is Nothing Then
            Call SetupLogXAxis(xAx, 10#, 20000#)
        End If

        ' Y軸タイトルの設定
        If Not chart.Axes(xlValue, xlPrimary).HasTitle Then
            chart.Axes(xlValue, xlPrimary).HasTitle = True
        End If
        chart.Axes(xlValue, xlPrimary).AxisTitle.Text = "Gain (dB)"
        chart.Axes(xlValue, xlPrimary).AxisTitle.Font.Size = fontSize + 2
        chart.Axes(xlValue, xlPrimary).AxisTitle.Font.Bold = False
        chart.Axes(xlValue, xlPrimary).AxisTitle.Font.Color = RGB(0, 0, 0)

        '=== Y軸（縦軸）のスケーリング設定 ===
        ' 主目盛り：10 dB刻み、補助目盛り：5 dB刻み
        ' 最小値：-40 dB、最大値：プロットの最大値を10の位で切り上げ
        Dim yMax As Double, yMaxCeil As Double
        yMax = GetChartMaxY(chart)
        yMaxCeil = WorksheetFunction.Ceiling(yMax, 10)
        With chart.Axes(xlValue)
            .MinimumScaleIsAuto = False
            .MaximumScaleIsAuto = False
            .MinimumScale = -40
            .MaximumScale = yMaxCeil
            .MajorUnit = 10
            .MinorUnit = 5
            .HasMinorGridlines = True
            .MajorTickMark = xlOutside
            .MinorTickMark = xlOutside
        End With

        ' 凡例の書式
        If chart.HasLegend Then
            With chart.Legend
                With .Font
                    .Size = fontSize - 2
                    .Bold = False
                    .Color = RGB(0, 0, 0)
                End With
                With .Format.Fill
                    .ForeColor.RGB = RGB(255, 255, 255)
                    .Transparency = 0.1
                End With
            End With
        End If

        '--- 線色と透明度をシリーズ番号に応じて設定（1→11で不透明度UP） ---
        Const BASE_R As Long = 10
        Const BASE_G As Long = 62
        Const BASE_B As Long = 85
        Const MAX_SERIES As Long = 11
        Const TRANS_START As Double = 0.65
        Const TRANS_END   As Double = 0.05

        Dim s As series
        Dim idx As Long
        Dim frac As Double
        Dim t As Double

        idx = 1
        For Each s In chart.SeriesCollection
            With s.Format.Line
                .Visible = msoTrue
                .Weight = 1.75
                .ForeColor.RGB = RGB(BASE_R, BASE_G, BASE_B)
                frac = WorksheetFunction.Min(idx - 1, MAX_SERIES - 1) / (MAX_SERIES - 1)
                t = TRANS_START + (TRANS_END - TRANS_START) * frac
                If t < 0 Then t = 0
                If t > 1 Then t = 1
                .Transparency = t
            End With
            idx = idx + 1
        Next s

        ' データラベルのフォント設定
        For Each series In chart.SeriesCollection
            If series.HasDataLabels Then
                series.DataLabels.Font.Size = fontSize
                series.DataLabels.Font.Bold = False
                series.DataLabels.Font.Color = RGB(0, 0, 0)
            End If
        Next series

        '=== グラフタイトルは使用しない ===
        ' 上部タイトルを無効化
        chart.HasTitle = False
        ' 既存の下部タイトル（BottomTitle）があれば削除
        Dim shp As Shape
        For Each shp In chart.Shapes
            If shp.Name = "BottomTitle" Then
                shp.Delete
                Exit For
            End If
        Next shp
        ' 下端が詰まりすぎないように、X軸ラベルの更に下に最低パディングを確保
        ' 第2引数は確保したい下端パディング（ポイント）。フォントサイズ由来の動的量 + 余白を足す。
        Call EnsureBottomPadding(chart, fontSize * 0.9 + 8, 0)
    Next chartObj
End Sub

' 下部余白を回収してプロットエリア高さを拡大する
Private Sub ReclaimBottomSpace(ch As chart)
    On Error Resume Next
    Dim avail As Single
    avail = ch.ChartArea.Height - (ch.PlotArea.Top + ch.PlotArea.Height)
    If avail > 0 Then
        ch.PlotArea.Height = ch.PlotArea.Height + avail
    End If
End Sub

Private Sub EnsureBottomPadding(ch As chart, paddingPts As Single, Optional minPlotHeight As Single = 60)
    On Error Resume Next
    
    Dim axisBottom As Single
    Dim currentGap As Single
    Dim diff As Single
    
    ' X軸ラベル下端の位置と、ChartArea下端とのギャップを取得
    axisBottom = ch.Axes(xlCategory).Top + ch.Axes(xlCategory).Height
    currentGap = ch.ChartArea.Height - axisBottom
    
    ' 目標との差分（+：余白が不足、-：余白が過大）
    diff = paddingPts - currentGap
    
    ' 許容誤差内なら調整不要
    If Abs(diff) <= 0.5 Then GoTo CleanExit
    
    If diff > 0 Then
        ' --- 余白が足りない：プロットエリアを上に押し上げる（高さを縮める優先） ---
        Dim deltaUp As Single
        deltaUp = diff
        
        If ch.PlotArea.Height > deltaUp + minPlotHeight Then
            ch.PlotArea.Height = ch.PlotArea.Height - deltaUp
        Else
            ' 最低高さを割る分はTopを上に上げて確保
            Dim needMoveUp As Single
            needMoveUp = deltaUp - (ch.PlotArea.Height - minPlotHeight)
            ch.PlotArea.Height = minPlotHeight
            ch.PlotArea.Top = Application.Max(0, ch.PlotArea.Top - needMoveUp)
        End If
    
    Else
        ' --- 余白が多すぎる：プロットエリアを下方向へ拡げる／下へ移動 ---
        Dim needReduce As Single
        needReduce = -diff   ' 減らしたいギャップ量
        
        ' 1) まず高さを伸ばしてギャップを詰める
        Dim freeBottom As Single
        Dim grow As Single
        freeBottom = ch.ChartArea.Height - (ch.PlotArea.Top + ch.PlotArea.Height)
        grow = Application.Min(needReduce, Application.Max(0, freeBottom - 1))
        If grow > 0 Then
            ch.PlotArea.Height = ch.PlotArea.Height + grow
            needReduce = needReduce - grow
        End If
        
        ' 2) まだ余るなら、プロットエリアを下へ移動
        If needReduce > 0 Then
            freeBottom = ch.ChartArea.Height - (ch.PlotArea.Top + ch.PlotArea.Height)
            Dim moveDown As Single
            moveDown = Application.Min(needReduce, Application.Max(0, freeBottom - 1))
            If moveDown > 0 Then
                ch.PlotArea.Top = ch.PlotArea.Top + moveDown
            End If
        End If
    End If
    
CleanExit:
    On Error GoTo 0
End Sub




Option Explicit

Public Sub ExportAllChartsAsPng()
    Dim basePath As String
    basePath = ThisWorkbook.Path
    If Len(basePath) = 0 Then
        MsgBox "このブックを一度保存してください（保存先フォルダが取得できません）。", vbExclamation
        Exit Sub
    End If
    
    ' フォルダ名：images_MM_dd_hh-mm-ss（:は使えないため-に置換）
    Dim folderName As String
    folderName = "images_" & Format(Now, "MM_dd_hh-mm-ss")
    Dim outDir As String
    outDir = basePath & Application.PathSeparator & folderName
    
    On Error Resume Next
    MkDir outDir
    If Err.Number <> 0 Then
        On Error GoTo 0
        MsgBox "出力フォルダの作成に失敗しました：" & vbCrLf & outDir, vbCritical
        Exit Sub
    End If
    On Error GoTo 0

    Application.ScreenUpdating = False
    Application.EnableEvents = False
    Application.DisplayAlerts = False

    ' ?? ここで全シートの表示倍率を150%に揃える ??
    Call SetAllSheetsZoom(150)

    Dim savedCount As Long
    savedCount = 0
    
    ' ワークシート上の埋め込みグラフ（ChartObject）
    Dim ws As Worksheet
    For Each ws In ThisWorkbook.Worksheets
        If ws.ChartObjects.Count > 0 Then
            Dim chObj As ChartObject
            Set chObj = ws.ChartObjects(1)
            If Not chObj Is Nothing Then
                Dim f1 As String
                f1 = outDir & Application.PathSeparator & "image_" & SanitizeForFileName(ws.Name) & ".png"
                On Error Resume Next
                chObj.chart.Export Filename:=f1, FilterName:="PNG"
                If Err.Number = 0 Then
                    savedCount = savedCount + 1
                Else
                    MsgBox "保存に失敗: " & f1, vbExclamation
                    Err.Clear
                End If
                On Error GoTo 0
            End If
        End If
    Next ws
    
    ' グラフシート（Chartsコレクション）
    Dim cs As chart
    For Each cs In ThisWorkbook.Charts
        Dim f2 As String
        f2 = outDir & Application.PathSeparator & "image_" & SanitizeForFileName(cs.Name) & ".png"
        On Error Resume Next
        cs.Export Filename:=f2, FilterName:="PNG"
        If Err.Number = 0 Then
            savedCount = savedCount + 1
        Else
            MsgBox "保存に失敗: " & f2, vbExclamation
            Err.Clear
        End If
        On Error GoTo 0
    Next cs

    Application.DisplayAlerts = True
    Application.EnableEvents = True
    Application.ScreenUpdating = True
    
    MsgBox "完了しました。" & vbCrLf & _
           "保存先: " & outDir & vbCrLf & _
           "保存枚数: " & savedCount, vbInformation
End Sub

Private Sub SetAllSheetsZoom(ByVal pct As Long)
    Dim curSheet As Object
    Set curSheet = ActiveSheet

    ' ワークシート
    Dim ws As Worksheet
    For Each ws In ThisWorkbook.Worksheets
        If ws.Visible = xlSheetVisible Then
            ws.Activate
            ActiveWindow.Zoom = pct
        End If
    Next ws
    
    ' グラフシート
    Dim cs As chart
    For Each cs In ThisWorkbook.Charts
        If cs.Visible = xlSheetVisible Then
            cs.Activate
            ActiveWindow.Zoom = pct
        End If
    Next cs

    ' 元のシートに戻す
    If Not curSheet Is Nothing Then
        On Error Resume Next
        curSheet.Activate
        On Error GoTo 0
    End If
End Sub

Private Function SanitizeForFileName(ByVal s As String) As String
    Dim badChars As Variant
    badChars = Array("\", "/", ":", "*", "?", """", "<", ">", "|")
    Dim i As Long
    For i = LBound(badChars) To UBound(badChars)
        s = Replace$(s, CStr(badChars(i)), "_")
    Next i
    s = Trim$(s)
    Do While Len(s) > 0 And Right$(s, 1) = "."
        s = Left$(s, Len(s) - 1)
    Loop
    If Len(s) = 0 Then s = "Sheet"
    SanitizeForFileName = s
End Function


Option Explicit

Public Sub A1_1x()
    Dim curSheet As Object
    Set curSheet = ActiveSheet

    Application.ScreenUpdating = False
    Application.EnableEvents = False
    Application.DisplayAlerts = False

    On Error GoTo CleanExit

    ' --- ワークシート：ズーム=100%、A1をアクティブにしてスクロール ---
    Dim ws As Worksheet
    For Each ws In ThisWorkbook.Worksheets
        If ws.Visible = xlSheetVisible Then
            ws.Activate
            On Error Resume Next
            ActiveWindow.Zoom = 100
            On Error GoTo 0
            Application.GoTo Reference:=ws.Range("A1"), Scroll:=True
        End If
    Next ws

    ' --- グラフシート：ズーム=100%（A1は存在しないため未設定） ---
    Dim cs As chart
    For Each cs In ThisWorkbook.Charts
        If cs.Visible = xlSheetVisible Then
            cs.Activate
            On Error Resume Next
            ActiveWindow.Zoom = 100
            On Error GoTo 0
        End If
    Next cs

CleanExit:
    ' 元のシートに戻る
    If Not curSheet Is Nothing Then
        On Error Resume Next
        curSheet.Activate
        On Error GoTo 0
    End If

    Application.DisplayAlerts = True
    Application.EnableEvents = True
    Application.ScreenUpdating = True
End Sub

Option Explicit

Public Sub DuplicateLastSheetFive()
    Dim src As Object ' Worksheet または Chart を許容
    Dim i As Long
    Dim wasScreenUpdating As Boolean, wasEnableEvents As Boolean, wasDisplayAlerts As Boolean
    
    If ThisWorkbook.Sheets.Count = 0 Then
        MsgBox "シートが存在しません。", vbExclamation
        Exit Sub
    End If
    
    Set src = ThisWorkbook.Sheets(ThisWorkbook.Sheets.Count) ' 最後尾のシートを取得
    
    ' 動作の安定化
    wasScreenUpdating = Application.ScreenUpdating
    wasEnableEvents = Application.EnableEvents
    wasDisplayAlerts = Application.DisplayAlerts
    Application.ScreenUpdating = False
    Application.EnableEvents = False
    Application.DisplayAlerts = False
    
    On Error GoTo ERR_HANDLER
    
    ' 5枚複製して最後尾へ
    For i = 1 To 5
        src.Copy After:=ThisWorkbook.Sheets(ThisWorkbook.Sheets.Count)
        ' ここでExcelが自動的にシート名を付けます（例：「Sheet1 (2)」など）
        ' 独自の命名をしたい場合は、以下の例を参考に:
        ' ThisWorkbook.Sheets(ThisWorkbook.Sheets.Count).Name = src.Name & "_copy" & i
    Next i
    
    GoTo CLEANUP

ERR_HANDLER:
    MsgBox "複製中にエラーが発生しました: " & Err.Description, vbCritical

CLEANUP:
    Application.DisplayAlerts = wasDisplayAlerts
    Application.EnableEvents = wasEnableEvents
    Application.ScreenUpdating = wasScreenUpdating
End Sub

Option Explicit

Public Sub ImportCsvsIntoTemplateCopies()
    Dim basePath As String, csvName As String
    Dim tpl As Worksheet, outWS As Worksheet
    Dim wasScreenUpdating As Boolean, wasEnableEvents As Boolean, wasDisplayAlerts As Boolean, wasCalc As XlCalculation
    
    Set tpl = Nothing
    On Error Resume Next
    Set tpl = ThisWorkbook.Worksheets("Template")
    On Error GoTo 0
    If tpl Is Nothing Then
        MsgBox "Template シートが見つかりません。", vbExclamation
        Exit Sub
    End If
    
    basePath = ThisWorkbook.Path
    If Len(basePath) = 0 Then
        MsgBox "このブックを一度保存してください（保存先フォルダが取得できません）。", vbExclamation
        Exit Sub
    End If
    
    ' 高速化
    wasScreenUpdating = Application.ScreenUpdating
    wasEnableEvents = Application.EnableEvents
    wasDisplayAlerts = Application.DisplayAlerts
    wasCalc = Application.Calculation
    Application.ScreenUpdating = False
    Application.EnableEvents = False
    Application.DisplayAlerts = False
    Application.Calculation = xlCalculationManual
    
    On Error GoTo CLEAN_FAIL
    
    csvName = Dir(basePath & Application.PathSeparator & "*.csv")
    If Len(csvName) = 0 Then
        MsgBox "フォルダ内にCSVが見つかりませんでした。" & vbCrLf & basePath, vbInformation
        GoTo CLEAN_EXIT
    End If
    
    Do While Len(csvName) > 0
        ' --- Template を複製して処理用シートを作成 ---
        tpl.Copy After:=ThisWorkbook.Sheets(ThisWorkbook.Sheets.Count)
        Set outWS = ActiveSheet
        On Error Resume Next
        outWS.Name = Left$(SanitizeSheetName(RemoveCsvExt(csvName)), 31) ' 31文字制限
        On Error GoTo 0
        
        ' --- CSVを開いて配列に読み込み ---
        Dim csvFullPath As String
        csvFullPath = basePath & Application.PathSeparator & csvName
        
        Dim wbCsv As Workbook, wsCsv As Worksheet
        Set wbCsv = Nothing: Set wsCsv = Nothing
        
        ' UTF-8(BOM 無/有) に広く対応するため OpenText を使用
        Workbooks.OpenText Filename:=csvFullPath, _
                           Origin:=65001, DataType:=xlDelimited, Comma:=True, _
                           Local:=False
        Set wbCsv = ActiveWorkbook
        Set wsCsv = wbCsv.Worksheets(1)
        
        Dim data As Variant
        data = wsCsv.UsedRange.Value ' 1-based 2次元配列
        
        ' --- データを索引化： index→(frequency→mag) ---
        Dim dictIdx As Object
        Set dictIdx = CreateObject("Scripting.Dictionary") ' key: index (String), item: Dictionary(freqKey → mag)
        
        Dim i As Long, nRow As Long
        Dim freq As Double, mag As Double
        Dim idx As Variant ' indexは数値のはずだがキー化時は文字列で統一
        Dim freqKey As String
        
        If IsArray(data) Then
            nRow = UBound(data, 1)
            
            ' ヘッダ行を想定（1行目）→ 2行目から読み込む
            For i = 2 To nRow
                If Not IsError(data(i, 1)) And Not IsError(data(i, 2)) And Not IsError(data(i, 4)) Then
                    If Len(data(i, 1)) > 0 And Len(data(i, 2)) > 0 And Len(data(i, 4)) > 0 Then
                        freq = CDbl(data(i, 1))          ' A列 frequency_Hz
                        mag = CDbl(data(i, 2))           ' B列 mag
                        idx = CStr(CLng(data(i, 4)))     ' D列 step_index（文字列キー化）
                        
                        ' indexごとの周波数→mag辞書を用意
                        If Not dictIdx.Exists(idx) Then
                            Set dictIdx(idx) = CreateObject("Scripting.Dictionary")
                        End If
                        freqKey = GetFreqKey(freq)       ' 浮動小数の一致誤差対策
                        dictIdx(idx)(freqKey) = mag
                    End If
                End If
            Next i
        End If
        
        ' --- 複製シートへ貼り付け ---
        ' 1) 行見出し（B2↓）： step_index = 0 の frequency を順番に並べる
        Dim listFreq As Variant, r As Long
        listFreq = BuildSortedFreqList(dictIdx, "0") ' index=0 の周波数リスト（並び保持）
        If IsEmpty(listFreq) Then
            ' index 0 がない場合は、最小のindexを使って列挙（保険）
            Dim fallbackIdx As String
            fallbackIdx = FirstKey(dictIdx)
            listFreq = BuildSortedFreqList(dictIdx, fallbackIdx)
        End If
        
        If Not IsEmpty(listFreq) Then
            ' 貼り付け
            outWS.Range("B2").Resize(UBound(listFreq) - LBound(listFreq) + 1, 1).Value = _
                ToVerticalRange(listFreq)
        End If
        
        ' 2) 列見出し（C1→）の index を読み取り、frequency×indexで mag を埋める
        Dim lastCol As Long, c As Long
        lastCol = outWS.Cells(1, outWS.Columns.Count).End(xlToLeft).Column
        If lastCol < 3 Then lastCol = 2 ' C列未満なら処理スキップ
        
        For c = 3 To lastCol
            Dim headerVal As Variant
            headerVal = outWS.Cells(1, c).Value
            
            If IsNumeric(headerVal) Then
                Dim idxKey As String
                idxKey = CStr(CLng(headerVal))
                
                If dictIdx.Exists(idxKey) Then
                    ' 各行の周波数に対する mag を充填
                    For r = LBound(listFreq) To UBound(listFreq)
                        freqKey = GetFreqKey(CDbl(listFreq(r)))
                        If dictIdx(idxKey).Exists(freqKey) Then
                            outWS.Cells(r - LBound(listFreq) + 2, c).Value = dictIdx(idxKey)(freqKey)
                        Else
                            ' 該当frequencyが無ければ空白（必要ならNA等）
                            outWS.Cells(r - LBound(listFreq) + 2, c).Value = vbNullString
                        End If
                    Next r
                Else
                    ' ヘッダーにある index にCSV側のデータが無い
                    ' → 空欄のまま（必要なら0を入れる等に変更可）
                End If
            End If
        Next c
        
        ' CSVを閉じる（保存しない）
        wbCsv.Close SaveChanges:=False
        
        ' 次のCSVへ
        csvName = Dir()
    Loop
    
    MsgBox "完了しました。", vbInformation
    GoTo CLEAN_EXIT

CLEAN_FAIL:
    MsgBox "エラー: " & Err.Description, vbCritical

CLEAN_EXIT:
    ' 復帰
    Application.Calculation = wasCalc
    Application.DisplayAlerts = wasDisplayAlerts
    Application.EnableEvents = wasEnableEvents
    Application.ScreenUpdating = wasScreenUpdating
End Sub

' ===== ヘルパー =====

Private Function GetFreqKey(ByVal f As Double) As String
    ' 周波数をキー化（丸め誤差対策：有効桁を揃えて文字列化）
    GetFreqKey = Format$(f, "0.##############") ' 必要に応じて桁数調整
End Function

Private Function BuildSortedFreqList(ByVal dictIdx As Object, ByVal idxKey As String) As Variant
    ' 指定 index の frequency キー一覧（キー順のまま＝CSVの順序）を Double 配列で返す
    ' Scripting.Dictionary は挿入順を保持する（現行Excel/VBAの実装前提）
    If Not dictIdx.Exists(idxKey) Then Exit Function
    Dim k As Variant, i As Long
    ReDim arr(1 To dictIdx(idxKey).Count) As Double
    i = 1
    For Each k In dictIdx(idxKey).Keys
        arr(i) = CDbl(k)
        i = i + 1
    Next k
    BuildSortedFreqList = arr
End Function

Private Function ToVerticalRange(ByVal arr As Variant) As Variant
    ' 1次元配列 → 縦ベクトル(2次元)にして返す
    Dim i As Long, lo As Long, hi As Long
    lo = LBound(arr): hi = UBound(arr)
    Dim m As Variant
    ReDim m(1 To hi - lo + 1, 1 To 1)
    For i = lo To hi
        m(i - lo + 1, 1) = arr(i)
    Next i
    ToVerticalRange = m
End Function

Private Function RemoveCsvExt(ByVal s As String) As String
    If LCase$(Right$(s, 4)) = ".csv" Then
        RemoveCsvExt = Left$(s, Len(s) - 4)
    Else
        RemoveCsvExt = s
    End If
End Function

Private Function SanitizeSheetName(ByVal s As String) As String
    Dim badChars As Variant, i As Long
    badChars = Array("/", "\", "[", "]", ":", "*", "?", """")
    For i = LBound(badChars) To UBound(badChars)
        s = Replace$(s, CStr(badChars(i)), "_")
    Next i
    ' 先頭・末尾のシングルクォート回避
    If Left$(s, 1) = "'" Then s = "_" & Mid$(s, 2)
    If Right$(s, 1) = "'" Then s = Left$(s, Len(s) - 1) & "_"
    If Len(Trim$(s)) = 0 Then s = "Sheet"
    SanitizeSheetName = s
End Function

Private Function FirstKey(ByVal dict As Object) As String
    Dim k As Variant
    For Each k In dict.Keys
        FirstKey = CStr(k)
        Exit Function
    Next k
    FirstKey = vbNullString
End Function


